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
    #mod = eval(cfg.model.arch)(**cfg.model.attrs).to(rank)
    mod = ProjSegModelUseSP(**cfg.model.args).to(rank)
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
        saver = SaveParams(mod, cfg.snapshot, cfg.model.args.base)

    # Train
    niter = 0
    rampup_pl = len(loader) * cfg.hyperparam.rampup_pl
    rampup_cl = len(loader) * cfg.hyperparam.rampup_cl
    get_rampup = lambda x_iter, x_rampup: 1 if x_rampup == 0 else (min(x_iter, x_rampup) / x_rampup)
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

            image, superpixel, label = data
            sp_mask = get_sp_mask(superpixel.to(rank), cfg.data.train.max_superpixel)
            logit, embed = mod(image.to(rank), sp_mask)

            # 1. pointly-supervised loss
            loss_pnt = criteria_ce(logit, label.to(rank))

            # 2. superpixel labels
            with torch.no_grad():
                sp_pred = average_sp(torch.softmax(logit.detach(), 1), sp_mask) # NCM
                sp_pred2d = (sp_pred @ sp_mask.flatten(2)).view(logit.shape)
                sp_label_conf2d, sp_label2d = sp_pred2d.max(1)
                sp_label2d[sp_label_conf2d < cfg.hyperparam.pl_threshold] = 255
            loss_pl = criteria_ce(logit, sp_label2d)

            # 2. contrastive loss
            with torch.no_grad():
                sp_label_conf, sp_label = sp_pred.max(1)
                sp_label[sp_label_conf < cfg.hyperparam.pl_threshold] = 255
                sp_label_oh = F.one_hot(sp_label.flatten(), 256)[..., :logit.shape[1]].float() # BC
                sp_label_valid = sp_label_oh.max(1, keepdim=True)[0] # B1
                sp_label_dot = sp_label_oh @ sp_label_oh.transpose(0,1) # BB
                sp_label_nonself = 1 - torch.eye(sp_label_dot.shape[0], dtype=torch.float32, device=rank)
                sp_label_ppair = (sp_label_dot * sp_label_valid) * sp_label_nonself # BB
                #sp_label_npair = (1 - sp_label_dot) * sp_label_valid # BB

            embed_flat = embed.transpose(1, 2).view(sp_label_oh.shape[0], embed.shape[1]) # BC
            embed_dot = (embed_flat @ embed_flat.transpose(0, 1)) / cfg.hyperparam.tau
            embed_logit = torch.log_softmax(embed_dot, 1)
            loss_cl = - (embed_logit * sp_label_ppair).sum(1)
            loss_cl = loss_cl / torch.clamp_min(sp_label_ppair.sum(1), 1)
            loss_cl = loss_cl.sum() / torch.clamp_min(sp_label_valid.sum(), 1)

            loss = cfg.hyperparam.lambda_ce * loss_pnt + \
                    (cfg.hyperparam.lambda_pl * get_rampup(niter, rampup_pl)) * loss_pl + \
                    (cfg.hyperparam.lambda_cl * get_rampup(niter, rampup_cl)) * loss_cl
            loss.backward()
            optimizer.step()
            niter += 1

            # Monitor
            if rank == 0:
                logger.update()
                if niter % cfg.log_freq == 0:
                    logger.push("Lall", loss.item())
                    logger.push("Lce", loss_pnt.item())
                    logger.push("Lpl", loss_pl.item())
                    logger.push("Lcl", loss_cl.item())
                    logger.push("rampup", get_rampup(niter, rampup_pl))
                    logger.flush(f"Epoch={epoch}, Iter=[{niter}/{total_iter}]")

                    vid = niter % image.shape[0]
                    demo = make_demo(
                            image[vid],
                            superpixel[vid],
                            label[vid],
                            sp_label2d[vid],
                            logit[vid]
                        )
                    imwrite(cfg.snapshot / "train_demo_preview.jpg", demo)
                    examples.append(demo)

        if rank == 0:
            saver.save(epoch)
            logger.info(f"Saved params to: {saver.filename}", 'y')

            imwrite(cfg.snapshot / "train_demo" / f"{cfg.model.args.base}-{epoch:04d}.jpg",
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
        mod = eval(cfg.model.args.base)(**cfg.model.attrs).to(rank)
        if is_dist:
            mod = nn.parallel.DistributedDataParallel(mod, device_ids=[rank],
                find_unused_parameters=False)
        params = cfg.snapshot / f"{cfg.model.args.base}-{cfg.schedule.num_epochs-1:04d}.params"
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
    cfg.snapshot = Path(args.snapshot) / cfg.model.args.base
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
