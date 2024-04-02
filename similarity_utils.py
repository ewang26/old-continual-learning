##############################################
### Image Similarity functions
##############################################

import numpy as np
from image_similarity_measures.quality_metrics import rmse, issm, psnr, ssim, uiq
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr



'''
Installation Notes: 
pip install pyfftw
pip install image-similarity-measures[speedups]

How to use:
(0) import any of the functions here that you want to use
(1) call vec_to_img on the vector of weights to convert to img
(2) then call amy of the similarity metrics on two images to get a floating pt value

See img_sim_tests.ipynb for an example
'''

def vec_to_img(vec):
    '''
    vec = 1d numpy array
    '''
    img_length = int(len(vec)**0.5)
    img = np.reshape(vec, (img_length, img_length))
    img = ((img - img.min()/(img.max()+1e-6))*256).astype(np.uint8)
    return img

def get_rmse(i1, i2):
    max_p = max(np.max(i1), np.max(i2))
    return rmse(i1, i2, max_p)

def get_issm(i1, i2):
    return issm(i1, i2)

def get_psnr(i1, i2):
    max_p = max(np.max(i1), np.max(i2))
    return psnr(i1, i2, max_p)

def get_ssim(i1, i2):
    max_p = max(np.max(i1), np.max(i2))
    return ssim(i1, i2, max_p)

def get_uiq(i1, i2):
    return uiq(i1, i2)