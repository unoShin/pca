import numpy as np

def PCA(img, num_ev):

    # if you want to fix the size of features as input
    #img = np.pad(img, 2, 'edge')

    # if you need accurate mean method, but here it is not important so we just use mean value for whole input
    img = img - np.mean(img)

    h, w = img.shape

    patch = np.zeros((25, (h - 4) * (w - 4)))

    for m in range(h - 4):
        for n in range(w - 4):
            patch[:, m * (w - 4) + n] = np.reshape(img[m:m + 5, n:n + 5], 25)

    # if you need accurate mean method, but here it is not important so we just use mean value for whole input
    #mu = np.mean(patch, axis=1)
    #print(mu.shape)
    #for j in range(25):
    #    patch[j] = patch[j] - mu[j]
    cov = np.dot(patch, patch.T)
    eigvals, eigvecs = np.linalg.eig(cov)

    Y = np.dot(eigvecs[:, 0:num_ev].T, patch)

    return Y, eigvecs[:, 0:num_ev]

def Recon(Y, eigvecs, h, w):
    X_est = np.dot(eigvecs, Y)
    X_fin = np.zeros((h, w))
    for m in range(h-4):
        for n in range(w-4):
            X_fin[m:m + 5, n:n + 5] = np.reshape(X_est[:, m * (w-4) + n], (5, 5))

    X_fin = np.reshape(X_fin, (h * w))

    return X_fin