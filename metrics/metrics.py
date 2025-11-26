import cv2
import numpy as np
from functools import partial

from metrics.quality import MSE_between_two_images, SSIM_between_two_img, LPIPS_between_two_img
from metrics.lfiq import LFIQ
from metrics.fic import FIC_min

_DEFAULT_JPG_QUALITY = 100
imwrite = partial(cv2.imwrite, params=[int(cv2.IMWRITE_JPEG_QUALITY), _DEFAULT_JPG_QUALITY])

def compute_metrics(opts,metrics, original_img, inverted_img, inverted_id, models, data_ids, swinface_embds,arcface_embds):
    """
    Update the dict metrics
    Args:
        opts: options
        metrics: dict of metric=[list]
        original_img: original image opened with PIL
        inverted_img: original image opened with PIL
        inverted_id: identifier of the inverted image
        models: loaded models to evaluate inversion
        data_ids: identifier for all dataset
        swinface_embds: embeddings for all original images
        arcface_embds: embeddings for all original images

    Returns: updated metrics for the given image
    """
    if opts.metrics.mse:
        # MSE comparaison
        mse = MSE_between_two_images(original_img, inverted_img)
        metrics["MSE"].append(mse)

    if opts.metrics.ssim:
        # SSIM comparison
        ssim = SSIM_between_two_img(original_img, inverted_img)
        metrics["SSIM"].append(ssim)

    if opts.metrics.lpips:
        # LPIPS comparison
        lpips = LPIPS_between_two_img(original_img, inverted_img, models["lpips"])
        metrics["LPIPS"].append(lpips)

    if opts.metrics.lfiq:
        # FIQA diff
        fiqa_diff = max(LFIQ(original_img, inverted_img, models["fiqa_trans"], models["fiqa"]),0.0)
        metrics["LFIQ"].append(fiqa_diff)

    if opts.metrics.fic_arcface:
        # ID - Arcface
        idd_min_arcface = FIC_min(inverted_img,inverted_id,arcface_embds,data_ids,models["arcface"],"arcface")
        metrics["FIC ARCFACE"].append(idd_min_arcface)

    if opts.metrics.fic_swinface:
        # ID - Swinface
        idd_min_swinface = FIC_min(inverted_img, inverted_id,swinface_embds,data_ids,models["swinface"],"swinface")
        metrics["FIC SWINFACE"].append(idd_min_swinface)

    return metrics

def average_metrics(metrics):
    """
    Args:
        metrics: dict with list of values (one for each metric)
    Returns: average for each metrics in the list
    """
    metrics_average = []
    for metric_name,metric_values in metrics.items():
        if metric_name != "FAC":
            average = sum(metric_values) / len(metric_values)
            metrics_average.append(average)
            print("AVERAGE FOR "+metric_name+" = ",average)

    """fiqa_changes = 0
    for i in metrics["FIQA"]:
        if i!= 0.0:
            fiqa_changes+=1
    fiqa_changes = (fiqa_changes / len(metrics[3])) * 100"""

    return metrics_average

def std_metrics(metrics):
    """
    Same as for average, but std
    """
    metrics_std = []
    for metric_name, metric in metrics.items():
        if metric_name != "FAC":
            metrics_std.append(np.std(metric))

    return metrics_std





















