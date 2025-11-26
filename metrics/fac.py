import numpy as np
from models.swinface_project.inference import inference, get_binary_classification

def diff_classif(vec_1,vec_2):
    """
    Args:
        vec_1: classification vector (binary) 1
        vec_2: classification vector (binary) 2

    Returns: list of differences (0 no diff, 1 for differences)
    """
    diff = []
    for i in range(len(vec_1)):
        if vec_1[i]!=vec_2[i]:
            diff.append(1)
        else:
            diff.append(0)
    return diff

def fac_img_i(model_swinface,original_img,inverted_img):
    """
    #Get facial attribute classification for original and reconstructed image, before computing the difference
    Args:
        model_swinface: loaded swinface model
        original_img: image opened with PIL
        inverted_img: image opened with PIL

    Returns: vector of n value indicating changes or not for the n attributes
    """

    classif_og = get_binary_classification(inference(model_swinface, original_img))
    classif_rec = get_binary_classification(inference(model_swinface, inverted_img))
    classif_changes = diff_classif(classif_og,classif_rec)

    return classif_changes

def fac(variant_name, list_changes_classif, nb_images):
    """
    Compute the fac metrics by averaging the attribute changes for each attribute over all images
    Args:
        variant_name: name of the folder containing inverted images (ie experiment name)
        list_changes_classif: list of list, for each image the changes or not over the n attributes
        nb_images: nb of image in dataset

    Returns:
        avg_attributes_changes: FAC (average)
        std_attributes_changes: FAC (std)
        sums_attribute_changes: list of the % of changes for each n attribute

    """


    # for each attribute n, do the sum of the attribute changes over all images (column)
    sums_attribute_changes = []
    for attribute_i in range(len(list_changes_classif[0])):
        sum_temp = 0
        for vec_difference in list_changes_classif:
            sum_temp += vec_difference[attribute_i]
        sums_attribute_changes.append((sum_temp / nb_images) * 100)



    #average over all n attributes (row)
    attributes_changes = []
    for sum_i in sums_attribute_changes:
        attributes_changes.append(sum_i)

    avg_attributes_changes = sum(attributes_changes) / len(sums_attribute_changes)
    std_attributes_changes = np.std(attributes_changes)

    sums_attribute_changes.insert(0, variant_name)

    return avg_attributes_changes, std_attributes_changes, sums_attribute_changes