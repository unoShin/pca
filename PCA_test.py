import scipy.misc
import numpy as np
from PCA import PCA, Recon
import cv2
import scipy.ndimage

img = scipy.misc.imread('butterfly_GT.bmp', flatten=True, mode='YCbCr').astype(np.float)

scale = 2
noise_factor = 20.

#img = scipy.ndimage.interpolation.zoom(img, (1./scale), prefilter=False)
#img = scipy.ndimage.interpolation.zoom(img, (scale/1.), prefilter=False)
#img = img + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=img.shape)

img = np.array(img, dtype=np.uint8)

scipy.misc.imsave('test/org.png', img)


img = np.array(img, dtype=np.float)
print(np.max(img))
print(np.min(img))

h, w = img.shape

num_out = 8

mu = np.mean(img)

Y, ev1 = PCA(img, num_out)

Y2 = np.zeros((num_out, num_out, (h - 8)*(w - 8)))
ev2 = np.zeros((num_out, 25, num_out))

for m in range(num_out):
    Y2[m], ev2[m] = PCA(np.reshape(Y[m], (h - 4, w - 4)), num_out)

for n in range(num_out):
    for j in range(num_out):
        out = np.reshape(Y2[n, j], (h - 8, w - 8))
        scipy.misc.imsave('test/pca_%d_%d.png' % (n, j), out)

# recon
X_est1 = np.zeros((num_out, (h-4) * (w-4)))
for k in range(num_out):
    X_est1[k] = Recon(Y2[k], ev2[k], h-4, w-4)

X_est = Recon(X_est1, ev1, h, w)

recon = np.reshape(X_est, (h, w))

#normalize
#X_fin = (recon - np.min(recon)) / (np.max(recon) - np.min(recon)) * (np.max(img) - np.min(img)) + np.min(img)
#X_fin = X_fin.astype(np.uint8)

#print(np.max(X_fin))
#print(np.min(X_fin))

#scipy.misc.imsave('test/recon.png', recon)
cv2.imwrite('test/recon.png', recon + mu)