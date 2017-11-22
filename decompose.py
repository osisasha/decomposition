# decomposition an masking of solar image
# mag  - магнитограмма
# cont - континуум, скорректированный за потемнение к краю
import sunpy.map
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

cont = sunpy.map.Map(
    'hmi.ic_nolimbdark_720s.20100504_000000_TAI.1.continuum.fits')
mag = sunpy.map.Map(
    'hmi.m_45s.20100504_000045_TAI.2.magnetogram.fits')
cont.peek()
mag.peek()

mag_qs = 10                  # 10 Gauss for QS
thr_plage = 3                # MF in plage is thr_plage times stronger than QS

x, y = np.meshgrid(*[np.arange(v.value) for v in mag.dimensions]) * u.pixel
hpc_coords = mag.pixel_to_world(x, y)
r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / mag.rsun_obs
cont.data[r > 1] = np.nan
mag.data[r > 1] = np.nan

mask_qs = np.ma.masked_less(np.abs(mag.data), mag_qs)
# masked array with Quiet Sun pixels,
# mask_qs = True where abs(mag.data) < mag_qs
cutoff_qs = np.nanmean(cont.data[mask_qs.mask])        # QS level

# sunspots excluded
# sub = np.ma.masked_greater(cont.data, 0.9 * cutoff_qs)
# dmin, dmax = cont.data[sub.mask].min(), cont.data[sub.mask].max()
# num_bins = cont.data.size  # num_bins = cont.data.size**(1/3) for plotting
# hist, bins = np.histogram(cont.data[sub.mask],
#                           bins=np.linspace(dmin, dmax, num_bins))
# x = (bins[:-1] + bins[1:]) / 2  # width = 0.7 * (bins[1] - bins[0])

# chist = np.cumsum(hist) / sum(hist)

# array = np.array(chist)
# values = (0.75, 0.97)
# idxs = np.searchsorted(array, values, side="left")
# prev_idx_is_less = ((idxs == len(array)) |
#                     (np.fabs(values - array[np.maximum(idxs - 1, 0)]) <
#                     np.fabs(values - array[np.minimum(idxs, len(array) - 1)])))
# idxs[prev_idx_is_less] -= 1
# cutoff_b, cutoff_f = x[idxs]
cutoff_b, cutoff_f = 1.0134667009759779, 1.0460999313327957

# creating decomposition mask
model_mask = np.zeros(cont.data.shape)

# IN
sub = np.ma.masked_inside(cont.data, 0.9 * cutoff_qs, cutoff_b)
n_in = model_mask[sub.mask].size
B_in = (np.nanmin(np.abs(mag.data[sub.mask])),
        np.nanmax(np.abs(mag.data[sub.mask])))
model_mask[sub.mask] = 1

# NW lane
sub = np.ma.masked_inside(cont.data, cutoff_b, cutoff_f)
n_nw = model_mask[sub.mask].size
B_nw = (np.nanmin(np.abs(mag.data[sub.mask])),
        np.nanmax(np.abs(mag.data[sub.mask])))
model_mask[sub.mask] = 2

# enhanced NW
sub = np.ma.masked_inside(cont.data, cutoff_f, 1.19 * cutoff_qs)
n_enw = model_mask[sub.mask].size
B_enw = (np.nanmin(np.abs(mag.data[sub.mask])),
         np.nanmax(np.abs(mag.data[sub.mask])))
model_mask[sub.mask] = 3

# Plage
sub1 = np.ma.masked_greater(np.abs(mag.data), thr_plage * mag_qs)
sub2 = np.ma.masked_inside(cont.data, 0.95 * cutoff_qs, cutoff_f)
sub = sub1.mask * sub2.mask
n_plage = model_mask[sub].size
B_plage = (np.nanmin(np.abs(mag.data[sub])),
           np.nanmax(np.abs(mag.data[sub])))
model_mask[sub] = 4

# Facula
sub2 = np.ma.masked_greater(cont.data, 1.01 * cutoff_qs)
sub = sub1.mask * sub2.mask
n_facula = model_mask[sub].size
B_facula = (np.nanmin(np.abs(mag.data[sub])),
            np.nanmax(np.abs(mag.data[sub])))
model_mask[sub] = 5

# penumbra
sub = np.ma.masked_inside(cont.data, 0.65 * cutoff_qs, 0.9 * cutoff_qs)
n_penumbra = model_mask[sub.mask].size
B_penumbra = (np.nanmin(np.abs(mag.data[sub.mask])),
              np.nanmax(np.abs(mag.data[sub.mask])))
model_mask[sub.mask] = 6

# umbra
sub = np.ma.masked_less(cont.data, 0.65 * cutoff_qs)
n_umbra = model_mask[sub.mask].size
B_umbra = (np.nanmin(np.abs(mag.data[sub.mask])),
           np.nanmax(np.abs(mag.data[sub.mask])))
model_mask[sub.mask] = 7

models = ['Intranetwork', 'Network', 'Enhanced network',
         'Plage' 'Facula', 'Penumbra', 'Umbra']
N = [n_in, n_nw, n_enw, n_plage, n_facula, n_penumbra, n_umbra]
B = [B_in, B_nw, B_enw, B_plage, B_facula, B_penumbra, B_umbra]

for i in range(len(N)):
    print('{model}: N of elements = {number}, abs(B) range: {Bmin} -- {Bmax}'.format(
        model=models[i], number=N[i], Bmin=B[i][0], Bmax=B[i][1]))

print('N total = ', sum(N))
print('elements at start = ', mag.data[r < 1].size)
print('difference = ', sum(N) - mag.data[r < 1].size)

#model_mask[r > 1] = np.nan
