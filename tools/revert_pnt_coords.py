from pathlib import Path

def revert_items(x):
    if not isinstance(x, str):
        return [revert_items(_x) for _x in x]
    x0, x1, x2 = x.split(',')
    return ','.join([x0, x2, x1])

def revert(src, dst):
    with open(src) as f:
        data = [x.strip().split(' ') for x in f.readlines()]

    data = [x[0:1] + revert_items(x[1:]) for x in data]
    data = [' '.join(x) for x in data]

    Path(dst).resolve().parents[0].mkdir(parents=True, exist_ok=True)
    with open(dst, 'w') as f:
        f.write("\n".join(data))

revert(
        '../scripts/point_labels/Cityscape_1point_0ignore_uniform.txt',
        '../resources/labels/cityscapes/prev_1point_0ignore_uniform.txt'
)

