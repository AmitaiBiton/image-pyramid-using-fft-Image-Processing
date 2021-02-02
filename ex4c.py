import numpy as np
import cv2
from matplotlib.pyplot import figure
figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
from matplotlib import pyplot as plt
# create filter and do FFT on image
def FFT(image):
    size = image.shape
    f1 = np.ones((size))
    f2 = np.ones((size))
    for i in range(10):
        f1[:, int(size[0] / 2) - 5 + i] = 0
        f2[int(size[0] / 2) - 5 + i, :] = 0
    f5 = f1 + f2
    f6 = np.zeros((size))
    f6[f5 == 1] = 0
    f6[f5 == 0] = 1

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    final =fshift*f6

    final = np.fft.ifftshift(final)
    final = np.fft.ifft2(final)
    final = abs(final)
    #final  = final.astype(np.uint64)
    return final


def gaussians(image):

    Gaussians = np.array(
        [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256

    clean_image = cv2.filter2D(image,cv2.CV_64F, Gaussians)
    #afterClean = clean_image[2:425, 2:421]
    #result = afterClean[:423, :419]
    return clean_image

def pyrDown(image):
    small_to_large_image_size_ratio = 0.5
    small_img = cv2.resize(image,  # original image
                           (0, 0),  # set fx and fy, not the final size
                           fx=small_to_large_image_size_ratio,
                           fy=small_to_large_image_size_ratio,
                           interpolation=cv2.INTER_NEAREST)
    small_img = FFT(small_img)
    return small_img
def pyrUp(image):
    small_to_large_image_size_ratio = 2.0
    large_img = cv2.resize(image,  # original image
                           (0, 0),  # set fx and fy, not the final size
                           fx=small_to_large_image_size_ratio,
                           fy=small_to_large_image_size_ratio,
                           interpolation=cv2.INTER_NEAREST)
    large_img =FFT(large_img)
    return large_img


def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels = 8):
    # assume mask is float32 [0,1]
    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = pyrDown(GA)
        GB = pyrDown(GB)
        GM = pyrDown(GM)
        gpA.append(np.float64(GA))
        gpB.append(np.float64(GB))
        gpM.append(np.float64(GM))

    # generate Laplacian Pyramids for A,B and masks
    # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpA = [gpA[num_levels - 1]]
    lpB = [gpB[num_levels - 1]]
    gpMr = [gpM[num_levels - 1]]
    for i in range(num_levels - 1, 0, -1):
        print(i)
        # Laplacian: subtarct upscaled version of lower
        # level from current level
        # to get the high frequencies
        LA = np.subtract(gpA[i - 1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i - 1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i - 1])  # also reverse the masks

        # Now blend images according to mask in each level
    LS = []
    for la, lb, gm in zip(lpA, lpB, gpMr):
        ls = la * gm + lb * (1.0 - gm)
        ls.dtype = np.float64
        LS.append(ls)
        # now reconstruct
    ls_ = LS[0]
    # ls_.dtype = np.float64
    for i in range(1, num_levels):
        print("LS" + str(i))
        ls_ = pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    return ls_
if  __name__ == "__main__":

    body = cv2.imread('unnamed.jpg' ,0)
    head  = cv2.imread('Men+Home+400x400.jpg' ,0)
    mask = cv2.imread('mask_.jpg',0 )
    mask = cv2.normalize(mask.astype('float64'),None,0.0,1.0,cv2.NORM_MINMAX)
    LPB = Laplacian_Pyramid_Blending_with_mask(head , body,  mask,5)
    LPB = LPB.astype(np.float64)
    LPB = gaussians(LPB)
    plt.subplot(2, 2, 1), plt.imshow(head, cmap="gray"), plt.title('Me')
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(body, cmap="gray"), plt.title('Friend')
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(mask, cmap="gray"), plt.title('Mask')
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(LPB, cmap="gray"), plt.title('Result')
    plt.xticks([]), plt.yticks([])
    plt.show()