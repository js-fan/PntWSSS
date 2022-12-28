import argparse
import yaml
from easydict import EasyDict
from pathlib import Path

from dist_utils import *
from model_utils import *
from core.model.vgg import *
from core.data import SimplePointDataset
from core.utils import *

from compute_iou import *

_SPPlaette = np.random.randint(0, 256, (1024, 3), np.uint8)

def make_demo(image, sp, label, plabel, logit):
    image = SimplePointDataset.denormalize(image)
    sp = _SPPlaette[sp.data.cpu().numpy().ravel() % 1024].reshape(sp.shape+(3,))
    label = CS.palette[label.data.cpu().numpy().ravel()].reshape(label.shape+(3,))
    plabel = CS.palette[plabel.data.cpu().numpy().ravel()].reshape(plabel.shape+(3,))
    pred = CS.palette[logit.argmax(0).data.cpu().numpy().ravel()].reshape(logit.shape[1:]+(3,))
    return imhstack([image, sp, label, plabel, pred], height=360)

def train(cfg):
    torch.manual_seed(cfg.seed)
    is_dist, rank = get_dist_info()
    dataset = SimplePointDataset(**cfg.data.train)
    loader, sampler = get_dataloader(dataset, cfg.schedule.batch_size)

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

        total_iter = len(loader) * cfg.schedule.num_epochs
        logger.set_total(total_iter)

        saver = SaveParams(mod, cfg.snapshot, cfg.model.arch)

    # Train
    niter = 0
    rampup_pl = len(loader) * cfg.hyperparam.rampup_pl
    get_rampup = lambda x_iter, x_rampup: 1 if x_rampup == 0 else (min(x_iter, x_rampup) / x_rampup)
    for epoch in range(cfg.schedule.num_epochs):
        if is_dist: sampler.set_epoch(epoch)

        lr = lr_scheduler.get(epoch)
        for lr_mult, param_group in zip(lr_mult_params.keys(), optimizer.param_groups):
            param_group['lr'] = lr * lr_mult
            if rank == 0:
                logger.info(f"Set lr={lr*lr_mult} for {len(param_group['params'])} params.", 'y')

        if rank == 0:
            logger.clear()
            examples = []

        mod.train()
        for data in loader:
            optimizer.zero_grad()

            image, superpixel, label = data
            logit = mod(image.to(rank))

            # 1. pointly-supervised loss
            loss_ce = criteria_ce(logit, label.to(rank))

            # 2. superpixel labels
            sp_mask = get_sp_mask(superpixel.to(rank), cfg.data.train.max_superpixel, logit.shape[-2:])
            with torch.no_grad():
                sp_pred = average_sp(torch.softmax(logit, 1), sp_mask, map2d=True)
                sp_label_conf, sp_label = sp_pred.max(1)
                sp_label[sp_label_conf < cfg.hyperparam.pl_threshold] = 255
            loss_pl = criteria_ce(logit, sp_label)

            # 3. manifold learning

            # 0. All
            loss = cfg.hyperparam.lambda_ce * loss_ce + \
                   cfg.hyperparam.lambda_pl * get_rampup(niter, rampup_pl) * loss_pl

            loss.backward()
            optimizer.step()
            niter += 1

            # Monitor
            if rank == 0:
                logger.update()
                if niter % cfg.log_freq == 0:
                    logger.push("Lall", loss.item())
                    logger.push("Lce", loss_ce.item())
                    logger.push("Lpl", loss_pl.item())
                    logger.push("rampup", get_rampup(niter, rampup_pl))
                    logger.flush(f"Epoch={epoch}, Iter=[{niter}/{total_iter}]")

                    vid = niter % image.shape[0]
                    demo = make_demo(
                            image[vid],
                            superpixel[vid],
                            label[vid],
                            sp_label[vid],
                            logit[vid]
                        )
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
    test_dataset = SimplePointDataset(**cfg.data.test, rtn_src=True)
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
