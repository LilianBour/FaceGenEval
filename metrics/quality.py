import torch
import numpy as np
import torch.nn.functional as F
from skimage.metrics import structural_similarity

def MSE_between_two_images(img_a, img_b):
    img_a = np.asfarray(img_a)/255.0
    img_b = np.asfarray(img_b)/255.0
    squared_diff = (img_a - img_b) ** 2
    summed = np.sum(squared_diff)
    num_pix = img_a.shape[0] * img_a.shape[1]  # img1 and 2 should have same shape
    err = summed / num_pix
    return err

def SSIM_between_two_img(img_a,img_b):
    #Open images
    img_a = np.asfarray(img_a.convert('L'))
    img_b = np.asfarray(img_b.convert('L'))

    #Get distance
    ssim_score = structural_similarity(img_a, img_b, multichannel=True, data_range=255)
    return ssim_score

def LPIPS_between_two_img(img_a,img_b,loss):
    #Open images
    img_a = np.asfarray(img_a,dtype=np.float32)
    img_b = np.asfarray(img_b,dtype=np.float32)

    # image should be RGB, IMPORTANT: normalized to [-1,1]
    normalized_input_a = (img_a - np.amin(img_a)) / (np.amax(img_a) - np.amin(img_a))
    img_a = 2 * normalized_input_a - 1
    normalized_input_b = (img_b - np.amin(img_b)) / (np.amax(img_b) - np.amin(img_b))
    img_b = 2 * normalized_input_b - 1

    #convert to tensor
    img_a = img_a.transpose()
    img_a = torch.from_numpy(img_a)
    img_a = F.interpolate(img_a, size=256)
    img_b = img_b.transpose()
    img_b = torch.from_numpy(img_b)
    img_b = F.interpolate(img_b, size=256)

    #Get distance
    distance = loss(img_a, img_b)

    if distance.item()=='nan':
        return "None"
    else:
        return distance.item()