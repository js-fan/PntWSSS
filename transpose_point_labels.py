
src = './resources/whats_the_point/train_aug_points_gtBackground.txt'
dst = './resources/whats_the_point/train_aug_points_gtBackground_xy.txt'

def transpose_item(x):
    L, y, x = x.split(',')
    return ','.join([L, x, y])

with open(src) as f:
    data = [x.strip().split(' ') for x in f.readlines()]

data = [line[0:1] + list(map(transpose_item, line[1:])) for line in data]
data = [' '.join(line) for line in data]
with open(dst, 'w') as f:
    f.write('\n'.join(data))
