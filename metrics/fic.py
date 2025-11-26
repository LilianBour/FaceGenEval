import math
import torch

import numpy as np
from PIL import Image

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from insightface.app import FaceAnalysis
from models.swinface_project.inference import inference as inference_swinface



def FIC_min(img,img_id,emb_list,id_list,model_loaded,model_name):
    """
    Args:
        img: img reconstructed, PIL
        emb_list: list of all embs of the dataset
        id_list: list of all identity of the dataset, matching emb_list
        model_loaded: loaded face reco model
        model_name: ie swinface or arcface

    Returns: min distance intra identity / min distance from reconstructed to imgs from the dataset
    """
    min_distance_inter = math.inf
    min_distance_intra = math.inf

    if model_name == "arcface":
        embd_1 = arcface_normalized_embedding(img, model_loaded)
    elif model_name == "swinface":
        output_1 = inference_swinface(model_loaded, img)
        with torch.no_grad():
            emb_1 = output_1["Recognition"][0].numpy()
        # normalize vectors with l2
        embd_1 = emb_1 / np.linalg.norm(emb_1)

    for embd_i in range(len(emb_list)):
        cos_sim = np.dot(embd_1, emb_list[embd_i][1])
        dist = 1 - cos_sim
        if img_id != id_list[emb_list[embd_i][0]] and dist < min_distance_inter:
                min_distance_inter = dist
        elif img_id == id_list[emb_list[embd_i][0]] and dist < min_distance_intra:
                min_distance_intra = dist
    return min_distance_intra/min_distance_inter



def embedding_list_for_data(list_images,model_loaded,model_name):
    """
    Args:
        list_img : list of original images names
        model_loaded: loaded face reco model
        model_name: ie swinface or arcface

    Returns: list of embds for original images
    """
    list_embds = []
    for img_name in list_images:
        img = Image.open(img_name)
        if model_name == "arcface":
            embd = arcface_normalized_embedding(img, model_loaded)
        elif model_name == "swinface":
            output_1 = inference_swinface(model_loaded, img)
            with torch.no_grad():
                emb_1 = output_1["Recognition"][0].numpy()
            # normalize vectors with l2
            embd = emb_1 / np.linalg.norm(emb_1)
        list_embds.append([img_name.split("/")[-1], embd])
    return list_embds


###################ARCFACE#########################
def analyze_faces(face_analysis: FaceAnalysis, img_data: np.ndarray, det_size=(704, 704)):
    # NOTE: try detect faces, if no faces detected, lower det_size until it does
    detection_sizes = [None] + [(size, size) for size in range(704, 256, -64)] + [(256, 256)]

    for size in detection_sizes:
        faces = face_analysis.get(img_data, det_size=size)
        if len(faces) > 0:
            return faces

    return []

def arcface_normalized_embedding(img_1,arcface_model):
    """
    Compute the arcface embedding of a face, modified version of the code used in icfegm_clean
        using image size instead of search for faces by incrementation of reducing the size by 32 until one face remains
        based on the assumption that only one face per image
    Args:
        img_1: img, PIL
    Returns:  arcface embedding
    """
    #Convert to opencv format
    img_1 = np.array(img_1)

    # INPUT : Need square image as input with size as multiple of 32
    # Get images sizes
    faces_1 = analyze_faces(arcface_model,img_1,det_size=(704, 704))
    #faces_1 = arcface_model.get(img_1)
    #print(faces_1)
    emb_1 = faces_1[0]["embedding"]

    #normalize vectors with l2
    emb_1_l2 = emb_1/np.linalg.norm(emb_1)

    return emb_1_l2

def compare_two_faces_arcface(img1,img2, arcface_model):
    """
    Cos distance between arcface imgs
    Args:
        img1: img 1, PIL
        img2: img 2, PIL

    Returns: cos distance between two embeddings

    """
    embd_1 = arcface_normalized_embedding(img1, arcface_model)
    embd_2 = arcface_normalized_embedding(img2, arcface_model)

    # Cosine similarity
    cos_sim = np.dot(embd_1, embd_2)
    cos_distance = 1 - cos_sim

    return cos_distance