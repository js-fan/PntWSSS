import argparse
import yaml
from easydict import EasyDict
from pathlib import Path

from dist_utils import *
from core.model.vgg import *
from core.data import SimplePointDataset
from core.utils import *

from compute_iou import *

def _as_list(x):
    return [x] if not isinstance(x, (list, tuple)) else list(x)

def _strip_str(x):
    return _strip_str(x[0]) if not isinstance(x, str) else x

def _may_resize(data, target, mode='bilinear'):
    if isinstance(target, torch.Tensor):
        target = target.shape[-2:]
    assert len(target) == 2, target
    curr_size = data.shape[-2:]
    if (curr_size[0] == target[0]) and (curr_size[1] == target[1]):
        return data

    expand = data.ndim == 3
    if expand:
        data = data.unsqueeze(1)
    data = F.interpolate(data, target, mode=mode, align_corners=True if mode =='bilinear' else None)
    if expand:
        data = data.squeeze(1)
    return data

def criteria_ce(logit, label):
    loss = F.cross_entropy(logit, label, ignore_index=255, reduction='sum') / \
        torch.clamp_min((label < 255).float().sum(), 1e-5)
    return loss

def criteria_focal(logit, label, alpha=1, gamma=2):
    label_oh = F.one_hot(label, 256)[..., :logit.shape[1]].permute(0, 3, 1, 2).float()
    p = torch.softmax(logit, 1)
    w = alpha * ( (label_oh * (1 - p)).sum(1) )**gamma

    loss = F.cross_entropy(logit, label, ignore_index=255, reduction='none')
    loss = (loss * w).sum() / w.detach().sum()#torch.clamp_min((label < 255).float().sum(), 1e-5)
    return loss

def criteria_bal(logit, label, gamma=1):
    label_oh = F.one_hot(label, 256)[..., :logit.shape[1]].permute(0, 3, 1, 2).float()
    w = ((1 / torch.clamp_min(label_oh.sum((2, 3), keepdim=True), 1))**gamma * label_oh).sum(1)

    loss = F.cross_entropy(logit, label, ignore_index=255, reduction='none')
    loss = (loss * w).sum() / w.sum()
    return loss

def make_demo(image, label, logit):
    image = SimplePointDataset.denormalize(image)
    label = CS.palette[label.data.cpu().numpy().ravel()].reshape(label.shape+(3,))
    pred = CS.palette[logit.argmax(0).data.cpu().numpy().ravel()].reshape(logit.shape[1:]+(3,))
    return imhstack([image, label, pred], height=360)

@torch.no_grad()
def _get_superpixel_scores(logit_seg, superpixel, max_sp_num):
    device = logit_seg.device
    softmax = torch.softmax(logit_seg, 1)
    n, c, h, w = softmax.shape

    sp = superpixel.to(device)
    sp_oh = F.one_hot(sp, max_sp_num+1)[..., :max_sp_num].permute(0, 3, 1, 2).float() # nmhw
    sp_oh = _may_resize(sp_oh, softmax)

    sp_oh_flat = sp_oh.flatten(2)
    sp_probs = sp_oh_flat @ softmax.flatten(2).transpose(1, 2)
    sp_probs = sp_probs / torch.clamp_min(sp_oh_flat.sum(2, keepdim=True), 1e-5) # nmc
    sp_probs_2d = (sp_probs.transpose(1, 2) @ sp_oh_flat).view(n, c, h, w)
    return sp_probs_2d, sp_probs, sp_oh

def train(cfg):
    torch.manual_seed(cfg.seed)
    is_dist, rank = get_dist_info()
    train_dataset = SimplePointDataset(**cfg.data.train)
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

            image, label = data

            logit_seg = mod(image.to(rank))
            label = _may_resize(label.float(), logit_seg, 'nearest').long().to(rank)
            loss = criteria_ce(logit_seg, label)

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

        name = Path(_strip_str(others[-1])).name.rsplit('.', 1)[0]
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
