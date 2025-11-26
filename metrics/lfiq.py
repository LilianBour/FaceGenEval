import torch
import numpy as np

def FIQA(img,trans,model):
    """
    Compute FIQA score of img
    Args:
        img: image, PIL
        trans: transformation applied to images for FIQA
        model: model used to get FIQA scores (AI-KD(DiffQA(r))
    Returns: FIQA score
    """
    #Convert to numpy
    img = np.array(img)

    #Transformations
    img_transf = trans(img)

    #Unsqueeze to get similar to a batch
    img_unsq = img_transf.unsqueeze(0)

    #Move img to gpu
    img_gpu = img_unsq.to("cuda")

    #FIQA
    with torch.no_grad():
        _, preds = model(img_gpu)
    preds = preds.detach().squeeze().cpu().numpy()
    return preds

def LFIQ(img_og, img_rec, trans, model):
    """
    Compute the difference between the FIQA scores of original and reconstruction images
    Args:
        img_og: original image, PIL
        img_rec: reconstructed image, PIL
        trans: transformation applied to images for FIQA
        model: model used to get FIQA scores (AI-KD(DiffQA(r))

    Returns: Change in FIQA score

    """
    fiqa_og = FIQA(img_og,trans,model)
    fiqa_rec = FIQA(img_rec,trans,model)
    return fiqa_og-fiqa_rec