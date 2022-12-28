import argparse
import yaml
from easydict import EasyDict
from pathlib import Path

from dist_utils import *
from model_utils import *
from core.model import ProjSegModel, ProjSegModelUseSP
from core.data import SimplePointDataset
from core.utils import *
from compute_iou import *

from core.model.deeplab_v3plus import DeeplabV3Plus

_SPPlaette = np.random.randint(0, 256, (1024, 3), np.uint8)
def auto_make_demo(img, *others):
    img = SimplePointDataset.denormalize(img)
    out = [img]
    for x in others:
        if x.ndim == 3: # logit
            res = CS.palette[x.argmax(0).data.cpu().numpy().ravel()].reshape(x.shape[1:]+(3,))
        elif x.ndim == 2:
            x = x.data.cpu().numpy()
            max_val = x.max()
            if (max_val == 255) or (max_val < 19): # label
                res = CS.palette[x.ravel()].reshape(x.shape+(3,))
            else: # superpixel
                res = _SPPlaette[x.ravel() % 1024].reshape(x.shape+(3,))
        else:
            raise RuntimeError(x.shape)
        out.append(res) 
    return imhstack(out, height=360)

def train(cfg):
    torch.manual_seed(cfg.seed)
    is_dist, rank = get_dist_info()
    dataset = SimplePointDataset(**cfg.data.train)
    loader, sampler = get_dataloader(dataset, cfg.schedule.batch_size)
    if cfg.schedule.get('num_epochs', None) is None:
        cfg.schedule.num_epochs = (cfg.schedule.num_iters + len(loader) - 1) // len(loader)

    # Model
    mod = eval(cfg.model.type)(**cfg.model.args).to(rank)
    mod.init_params(cfg.model.pretrained)

    lr_mult_params = mod.get_param_groups()

    param_groups = []
    for lr_mult, params in lr_mult_params.items():
        param_groups.append({'params': params, 'lr': cfg.schedule.lr * lr_mult})
    optimizer = torch.optim.SGD(
        param_groups,
        lr=cfg.schedule.lr,
        momentum=cfg.schedule.momentum,
        weight_decay=cfg.schedule.weight_decay)
    lr_scheduler = LrScheduler('poly', cfg.schedule.lr,
            {'power': 0.9, 'num_epochs': cfg.schedule.num_epochs})

    if is_dist:
        mod = nn.parallel.DistributedDataParallel(mod, device_ids=[rank],
            find_unused_parameters=False)

    # Monitor
    if rank == 0:
        logger = get_logger(cfg.snapshot / f"{cfg.time_stamp}.log")
        logger.log_config(cfg, 'b')
        total_iter = len(loader) * cfg.schedule.num_epochs
        logger.set_total(total_iter)
        saver = SaveParams(mod, cfg.snapshot, cfg.model.type, 3)

    # Train
    niter = 0
    for epoch in range(cfg.schedule.num_epochs):
        if is_dist: sampler.set_epoch(epoch)
        mod.train()

        lr = lr_scheduler.get(epoch)
        for lr_mult, param_group in zip(lr_mult_params.keys(), optimizer.param_groups):
            param_group['lr'] = lr * lr_mult
            if rank == 0:
                logger.info(f"Set lr={lr*lr_mult} for {len(param_group['params'])} params.", 'y')
        if rank == 0:
            logger.clear()
            examples = []

        for data in loader:
            optimizer.zero_grad()
            
            image, label = data
            logit, logit_aux = mod(image.to(rank))

            loss_main = criteria_ce(logit, label.to(rank))
            loss_aux = criteria_ce(logit_aux, label.to(rank))

            loss = loss_main + loss_aux * cfg.hyperparam.aux_weight
            loss.backward()
            optimizer.step()
            niter += 1

            # Monitor
            if rank == 0:
                logger.update()
                if niter % cfg.log_freq == 0:
                    logger.push("Lall", loss.item())
                    logger.push("Lmain", loss_main.item())
                    logger.push("Laux", loss_aux.item())
                    logger.flush(f"Epoch={epoch}, Iter=[{niter}/{total_iter}]")

                    vid = niter % image.shape[0]
                    demo = auto_make_demo(image[vid], label[vid], logit[vid], logit_aux[vid])
                    imwrite(cfg.snapshot / "train_demo_preview.jpg", demo)
                    examples.append(demo)

        if rank == 0:
            saver.save(epoch)
            logger.info(f"Saved params to: {saver.filename}", 'y')

            imwrite(cfg.snapshot / "train_demo" / f"{cfg.model.type}-{epoch:04d}.jpg",
                imvstack(examples) )

        if ((epoch + 1) == cfg.schedule.num_epochs) or \
           ((epoch + 1) % cfg.schedule.test_epoch_interval == 0):
            test(cfg, mod)

def test(cfg, mod=None):
    is_dist, rank = get_dist_info()
    test_dataset = SimplePointDataset(**cfg.data.test, rtn_src=True)
    test_loader, _ = get_dataloader(test_dataset,
        dist.get_world_size() if is_dist else 1,
        shuffle=False, drop_last=False, num_workers=1
    )

    if mod is None:
        mod = eval(cfg.model.type)(**cfg.model.args).to(rank)
        if is_dist:
            mod = nn.parallel.DistributedDataParallel(mod, device_ids=[rank],
                find_unused_parameters=False)
        params = cfg.snapshot / f"{cfg.model.type}-{cfg.schedule.num_epochs-1:04d}.params"
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

        name = Path(strip_str(others[-1])).name.rsplit('.', 1)[0]
        cv2.imwrite(str(save_dir / f"{name}.png"), pred_label)

    if is_dist:
        dist.barrier()

    #TODO: rewrite the following hardcode for cs.
    if rank == 0:
        compute_cs_iou(str(save_dir), cfg.data.test.data_roots[1], logger=logger)

def main(args):
    with open(args.config) as f:
        cfg = EasyDict(yaml.safe_load(f))

    cfg.config = args.config
    cfg.seed = np.random.randint(0, 65536)
    cfg.snapshot = Path(args.snapshot) / cfg.model.type
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
