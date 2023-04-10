import sys
sys.path.append('..')
import numpy as np

def adjust(pic, mask ):
    base_line = 3 * np.random.rand(1) + 11.0
    d0 = pic * mask[:, :, 0]
    d1 = pic * mask[:, :, 1]

    d0_mean = (d0.sum(axis=-1).sum(axis=-1)) / (mask[:, :, 0].sum(axis=-1).sum(axis=-1))
    d1_mean = (d1.sum(axis=-1).sum(axis=-1)) / (mask[:, :, 1].sum(axis=-1).sum(axis=-1))
    sing_db = 20*np.log10(d1_mean/d0_mean)

    alpha = np.power(10.0, (sing_db / 20.0 - base_line / 20.0))

    temp = pic*(mask[:, :, 0])*alpha
    if temp.max()>1:
        temp[temp>1] = pic[temp>1]
    pic = temp + pic*(1-mask[:, :, 0])
    return pic

