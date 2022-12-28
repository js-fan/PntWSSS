import argparse
import yaml
from easydict import EasyDict
from pathlib import Path
from train_scripts import *
from core.data import SimplePointDataset
from core.data.VOC import VOCPointDataset
from core.model import *
from torch.optim import *
from compute_iou import *

def _strip_str(x):
    return _strip_str(x[0]) if not isinstance(x, str) else x

def train(cfg):
    torch.manual_seed(cfg.seed)
    is_dist, rank, world_size = get_dist_info()
    train_one_step = eval(cfg.train_func)

    # Data
    dataset = SimplePointDataset(**cfg.data.train)
    # dataset = VOCPointDataset(
    #     '/home/junsong_fan/diskf/data/VOC2012/JPEGImages',
    #     './resources/whats_the_point/train_aug_points_gtBackground.txt',
    #     None,
    #     'train_aug',
    #     [321, 321],
    #     rand_crop=True, rand_mirror=True, rand_scale=(0.5, 1.5),
    #     downsample_label=8,
    #     superpixel_root='/home/junsong_fan/diskf/data/VOC2012/superpixel/mcg_png',
    #     return_image_label=True
    # )
    loader, sampler = get_dataloader(dataset, cfg.optimizer.batch_size)
    cfg.optimizer.niters_per_epoch = len(loader)

    # Monitor
    if rank == 0:
        monitor = get_monitor(cfg.snapshot / f"{cfg.time_stamp}.log")
        monitor.init_eta(cfg.optimizer.niters_per_epoch * cfg.optimizer.num_epochs)
        monitor.init_auto_log(cfg.log_freq, "Epoch={}")
        monitor.log_config(cfg, 'b')
        num_classes = {'voc': 21, 'cs': 19, 'ade': 150}[cfg.data.type.lower()]
        visualizer = SEGVisualizer(num_classes, eval(f"{cfg.data.type.upper()}.palette"))
    else:
        monitor = None

    # Model
    mod = eval(cfg.model.type)(**cfg.model.args).to(rank)
    if is_dist:
        mod = nn.parallel.DistributedDataParallel(mod, device_ids=[rank], find_unused_parameters=False)

    # Optimizer
    params_lr_mult = get_params_lr_mult(mod, cfg.model.get('params_lr_mult', None))
    if monitor:
        for mult, params in params_lr_mult.items():
            monitor.info(f"Set lr_mult={mult} for {len(params)} params.", 'y')
    param_groups = [{'params': params, 'lr': cfg.optimizer.args.lr*mult}
            for mult, params in params_lr_mult.items()]
    optimizer = eval(cfg.optimizer.type)(param_groups, **cfg.optimizer.args)
    scheduler = eval(cfg.optimizer.scheduler.type)(optimizer, **cfg.optimizer.scheduler.args)
    if rank == 0:
        saver = Saver(mod, cfg.snapshot / "checkpoints", cfg.model.type)

    # Train
    niter = 0
    for epoch in range(1, cfg.optimizer.num_epochs+1):
        if is_dist: sampler.set_epoch(epoch)
        if rank == 0: monitor.clear()
        mod.train()

        for data in loader:
            optimizer.zero_grad()
            loss, log_args, log_imgs = train_one_step(rank, mod, data, niter, cfg)
            loss.backward()
            optimizer.step()
            niter += 1

            if monitor is not None:
                if monitor.requires_log:
                    for k, v in log_args.items():
                        v = v.item() if isinstance(v, torch.Tensor) else v
                        monitor.push(k, v)
                    if log_imgs is not None:
                        monitor.push_image(visualizer(*log_imgs))
                monitor.step(epoch)
        scheduler.step()

        if rank == 0:
            saver.save(epoch)
            monitor.info(f"Saved params to: {saver}", 'y')
            monitor.flush_images()

        if (epoch == cfg.optimizer.num_epochs) or (epoch % cfg.test.test_epoch_interval == 0):
            test(cfg, mod)

def test(cfg, mod=None):
    is_dist, rank, world_size = get_dist_info()
    dataset = SimplePointDataset(**cfg.data.test, rtn_src=True)
    loader, _ = get_dataloader(dataset, world_size, shuffle=False, drop_last=False, num_workers=1)

    if mod is None:
        mod = eval(cfg.model.type)(**cfg.model.args).to(rank)
        if is_dist:
            mod = nn.parallel.DistributedDataParallel(mod, device_ids=[rank], find_unused_parameters=False)
        params = cfg.snapshot / f"{cfg.model.type}-{cfg.optimizer.num_epochs}.pth"
        params = torch.load(params, map_location=f"cuda:{rank}")
        mod.load_stats(params)
    mod.eval()

    # Test
    save_dir = cfg.snapshot / "results"
    save_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for image, *others in loader:
            assert (image.ndim == 4) and (image.shape[0] == 1), image.shape
            h, w = image.shape[-2:]
            image = image.to(rank)

            preds = []
            for scale in cfg.test.test_scales:
                img = F.interpolate(image, scale_factor=scale, mode='bilinear', align_corners=True)
                if cfg.test.mirror:
                    img_flip = torch.flip(img, (3,))
                    img = torch.cat([img, img_flip], dim=0)
                pred = mod(img)
                pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
                if cfg.test.mirror:
                    pred = (pred[0:1] + torch.flip(pred[1:2], (3,))) / 2
                preds.append(pred)

            preds = sum(preds) / len(preds)
            pred_label = preds[0].argmax(0).data.cpu().numpy().astype(np.uint8)

            name = Path(_strip_str(others[-1])).name.rsplit('.', 1)[0]
            cv2.imwrite(str(save_dir / f"{name}.png"), pred_label)

    if is_dist:
        dist.barrier()

    if rank == 0:
        logger = get_monitor(cfg.snapshot / f"{cfg.time_stamp}.log")
        eval(f"compute_{cfg.data.type.lower()}_iou")(str(save_dir), cfg.data.test.data_roots[1], logger=logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--gpus', type=str, default=0)
    parser.add_argument('--log-freq', type=int, default=50)
    parser.add_argument('--snapshot', type=str, default='./snapshot2')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = EasyDict(yaml.safe_load(f))
    cfg.config = args.config
    cfg.seed = np.random.randint(0, 65536)
    cfg.snapshot = Path(args.snapshot) / cfg.model.type
    cfg.log_freq = args.log_freq
    cfg.time_stamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')

    Runner(args.gpus, train).run(cfg)
