import argparse
import yaml
from easydict import EasyDict
from pathlib import Path
import tempfile


from dist_utils import *
from core.model.vgg import *
from core.data import PointAugDatasetWithSP
from core.utils import *

from compute_iou import *

def _as_list(x):
    return [x] if not isinstance(x, (list, tuple)) else list(x)

def _strip_str(x):
    return _strip_str(x[0]) if not isinstance(x, str) else x

def make_demo(image, label, logit):
    image = PointAugDatasetWithSP.denormalize(image)
    label = CS.palette[label.data.cpu().numpy().ravel()].reshape(label.shape+(3,))
    pred = CS.palette[logit.argmax(0).data.cpu().numpy().ravel()].reshape(logit.shape[1:]+(3,))
    return imhstack([image, label, pred], height=360)

def train(cfg):
    is_dist, rank = get_dist_info()
    train_dataset = PointAugDatasetWithSP(**cfg.data.train)
    train_loader, train_sampler = get_dataloader(train_dataset, cfg.schedule.batch_size)

    # Model
    mod = eval(cfg.model.arch)(**cfg.model.attrs).to(rank)
    pretrained = torch.load(cfg.model.pretrained, map_location=f"cuda:{rank}")
    mod.init_params(pretrained, cfg.seed)

    # Schedule
    lr_mult_params = mod.get_param_groups()
    param_groups = []
    for lr_mult, params in lr_mult_params.items():
        param_groups.append({'params': params, 'lr': cfg.schedule.lr * lr_mult})
    optimizer = torch.optim.SGD(
        param_groups,
        lr=cfg.schedule.lr,
        momentum=cfg.schedule.momentum,
        weight_decay=cfg.schedule.weight_decay
    )
    lr_scheduler = LrScheduler('poly', cfg.schedule.lr, {'power': 0.9, 'num_epochs': cfg.schedule.num_epochs})

    if is_dist:
        mod = nn.parallel.DistributedDataParallel(mod, device_ids=[rank],
            find_unused_parameters=False)

    # Monitor
    if rank == 0:
        logger = get_logger(cfg.snapshot / f"{cfg.time_stamp}.log")
        logger.log_config(cfg, 'b')

        total_iter = len(train_loader) * cfg.schedule.num_epochs
        logger.set_total(total_iter)

        saver = SaveParams(mod, cfg.snapshot, cfg.model.arch)

    # Train
    niter = 0
    for epoch in range(cfg.schedule.num_epochs):
        if is_dist: train_sampler.set_epoch(epoch)

        lr = lr_scheduler.get(epoch)
        for lr_mult, param_group in zip(lr_mult_params.keys(), optimizer.param_groups):
            param_group['lr'] = lr * lr_mult
            if rank == 0:
                logger.info(f"Set lr={lr*lr_mult} for {len(param_group['params'])} params.", 'y')

        if rank == 0:
            logger.clear()
            examples = []

        for data in train_loader:
            optimizer.zero_grad()

            image, superpixel, label = data

            logit_seg = mod(image.to(rank))
            label = F.interpolate(label.float().unsqueeze(1).to(rank), logit_seg.shape[-2:], mode='nearest').long().squeeze(1)
            loss = F.cross_entropy(logit_seg, label.to(rank), ignore_index=255)

            loss.backward()
            optimizer.step()
            niter += 1

            # Monitor
            if rank == 0:
                logger.update()
                if niter % cfg.log_freq == 0:
                    logger.push("loss", loss.item())
                    logger.flush(f"Epoch={epoch}, Iter=[{niter}/{total_iter}]")

                    vid = niter % image.shape[0]
                    demo = make_demo(image[vid], label[vid], logit_seg[vid])
                    imwrite(cfg.snapshot / "train_demo_preview.jpg", demo)
                    examples.append(demo)

        if rank == 0:
            saver.save(epoch)
            logger.info(f"Saved params to: {saver.filename}", 'y')

            imwrite(cfg.snapshot / "train_demo" / f"{cfg.model.arch}-{epoch:04d}.jpg",
                imvstack(examples) )

        if ((epoch + 1) == cfg.schedule.num_epochs) or \
           ((epoch + 1) % cfg.schedule.test_epoch_interval == 0):
            test(cfg, mod)

def test(cfg, mod=None):
    is_dist, rank = get_dist_info()
    test_dataset = PointAugDatasetWithSP(**cfg.data.test, rtn_src=True)
    test_loader, _ = get_dataloader(test_dataset,
        dist.get_world_size() if is_dist else 1,
        shuffle=False, drop_last=False, num_workers=1
    )

    if mod is None:
        mod = eval(cfg.model.arch)(**cfg.model.attrs).to(rank)
        if is_dist:
            mod = nn.parallel.DistributedDataParallel(mod, device_ids=[rank],
                find_unused_parameters=False)
        params = cfg.snapshot / f"{cfg.model.arch}-{cfg.schedule.num_epochs-1:04d}.params"
        params = torch.load(params, map_location=f"cuda:{rank}")
        mod.load_stats(params)
    mod.eval()

    if rank == 0:
        logger = get_logger(cfg.snapshot / f"{cfg.time_stamp}.log")

    # Test
    save_dir = cfg.snapshot / "results"
    save_dir.mkdir(parents=True, exist_ok=True)
    test_scales = [0.75, 1, 1.25]
    for data in test_loader:
        image, *others = data
        assert image.shape[0] == 1, image.shape

        image = image.to(rank)
        _, _, h, w = image.shape
        preds = []

        with torch.no_grad():
            for scale in test_scales:
                img = F.interpolate(image, scale_factor=scale, mode='bilinear', align_corners=True)
                img_flip = torch.flip(img, (3,))
                pred = mod(torch.cat([img, img_flip], dim=0))
                pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
                preds += [pred[0:1], torch.flip(pred[1:2], (3,))]

        preds = sum(preds) / len(preds)
        pred_label = preds.argmax(1)[0].data.cpu().numpy().astype(np.uint8)

        name = Path(_strip_str(others[-1])).name.rsplit('.', 1)[0]
        cv2.imwrite(str(save_dir / f"{name}.png"), pred_label)

    if is_dist:
        dist.barrier()

    #TODO: rewrite the following hardcode for cs.
    if rank == 0:
        compute_cs_iou(str(save_dir), cfg.data.test.image_root[1], logger=logger)

def main(args):
    with open(args.config) as f:
        cfg = EasyDict(yaml.safe_load(f))

    cfg.config = args.config
    cfg.seed = np.random.randint(0, 65536)
    cfg.snapshot = Path(args.snapshot) / cfg.model.arch
    cfg.log_freq = args.log_freq
    cfg.time_stamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')

    # Train
    Runner(args.gpus, train).run(cfg)

    # Test
    # Runner(args.gpus, test).run(args.cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str)
    parser.add_argument('--gpus', type=str, default=0)
    parser.add_argument('--log-freq', type=int, default=50)
    parser.add_argument('--snapshot', type=str, default='./work_dir')
    args = parser.parse_args()

    main(args)
