from dist_utils import *

def as_list(x):
    return [x] if not isinstance(x, (list, tuple)) else list(x)

def strip_str(x):
    return strip_str(x[0]) if not isinstance(x, str) else x

def resize(data, target, mode='bilinear'):
    size = target.shape[-2:] if isinstance(target, torch.Tensor) else target
    assert len(size) == 2, size
    curr_size = data.shape[-2:]
    if (curr_size[0] == size[0]) and (curr_size[1] == size[1]):
        return data

    expand = data.ndim == 3
    if expand:
        data = data.unsqueeze(1)
    assert data.ndim == 4, data.shape
    align_corners = {
            'bilinear': True,
            'nearest': None
    }[mode]
    data = F.interpolate(data.float(), size, mode=mode, align_corners=align_corners).to(data.dtype)
    if expand:
        data = data.squeeze(1)
    return data

def criteria_ce(logit, label, ignore_index=255):
    label = resize(label, logit, 'nearest')
    loss = F.cross_entropy(logit, label, ignore_index=ignore_index, reduction='sum')
    num_elements = torch.clamp_min((label < ignore_index).float().sum(), 1)
    return loss / num_elements

def criteria_bal_ce(logit, label, gamma=1, ignore_index=255):
    label_oh = F.one_hot(label, ignore_index+1)[..., :logit.shape[1]].permute(0,3,1,2).float()
    bal = ( (1 / torch.clamp_min(label_oh.sum((2,3), keepdim=True), 1))**gamma * label_oh ).sum(1)
    loss = F.cross_entropy(logit, label, ignore_index=ignore_index, reduction='none')
    loss = (loss * bal).sum() / bal.sum()
    return loss

@torch.no_grad()
def get_sp_mask(sp, max_sp_num, size=None):
    assert max_sp_num is not None
    if size is not None:
        sp = resize(sp, size, 'nearest')
    sp_oh = F.one_hot(sp, max_sp_num+1)[..., :max_sp_num].permute(0,3,1,2).float() # nmhw
    return sp_oh

def average_sp(data, sp, max_sp_num=None, map2d=False):
    assert data.ndim == 4, data.shape
    if sp.ndim == 3:
        sp = get_sp_mask(sp, max_sp_num, data.shape[-2:])

    sp_flat = sp.flatten(2)
    # NCM
    avg_res = (data.flatten(2) @ sp_flat.transpose(1,2)) / \
            torch.clamp_min(sp_flat.sum(2).unsqueeze(1), 1)

    if map2d:
        n, c= data.shape[:2]
        h, w = sp.shape[-2:]
        avg_res = (avg_res @ sp_flat).view(n, c, h, w)

    return avg_res

