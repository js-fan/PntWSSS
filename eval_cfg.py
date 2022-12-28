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

def eval_something(cfg, mod=None):
    cfg.data.test.name_prefix = './resources/whats_the_point/sets/train_aug.txt'
    is_dist, rank, world_size = get_dist_info()
    dataset = SimplePointDataset(**cfg.data.test, rtn_src=True)
    loader, _ = get_dataloader(dataset, world_size, shuffle=False, drop_last=False, num_workers=1)

    if mod is None:
        mod = eval(cfg.model.type)(**cfg.model.args).to(rank)
        params = cfg.snapshot / "checkpoints" / f"{cfg.model.type}-{cfg.optimizer.num_epochs}.pth"
        params = torch.load(params, map_location=f"cuda:{rank}")
        if is_dist:
            mod = nn.parallel.DistributedDataParallel(mod, device_ids=[rank], find_unused_parameters=False)
            if not list(params.keys())[0].startswith('module.'):
                params = {f'module.{k}': v for k, v in params.items()}
        else:
            if list(params.keys())[0].startswith('module.'):
                params = {k.replace('module.', ''): v for k, v in params.items()}
        mod.load_state_dict(params)
    mod.eval()

    # Test
    save_dir = cfg.snapshot / "cache" / "trainset"
    save_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for image, image_label, src in loader:
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
            preds_softmax = torch.softmax(preds, 1)
            preds_score, preds_label = preds_softmax.max(1)
            preds_entropy = - (preds_softmax * torch.log_softmax(preds, 1)).sum(1)

            preds_label = preds_label[0].data.cpu().numpy().astype(np.uint8)
            preds_score = preds_score[0].data.cpu().numpy().astype(np.float16)
            preds_entropy = preds_entropy[0].data.cpu().numpy().astype(np.float16)
            image_label = image_label[0].data.cpu().numpy().astype(np.uint8)

            name = Path(_strip_str(src)).name.rsplit('.', 1)[0]
            cv2.imwrite(str(save_dir / f'{name}.png'), preds_label)
            np.save(str(save_dir / f'{name}_score.npy'), preds_score)
            np.save(str(save_dir / f'{name}_entropy.npy'), preds_entropy)
            np.save(str(save_dir / f'{name}_imgLabel.npy'), image_label)

    if is_dist:
        dist.barrier()

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

    Runner(args.gpus, eval_something).run(cfg)
