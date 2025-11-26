import os
import pickle
import argparse
import torch 
from torch.utils.data import DataLoader
from tqdm import tqdm
from FIQA.utils import *
from FIQA.dataset import *
import csv


@torch.no_grad()
def inference(args,path_to_main_folder,path_for_results_csv,results_csv_name_FIQA):

    # Seed all libraries to ensure consistency between runs.
    seed_all(args.base.seed)

    # Load the trained AI-KD model and construct the transformation
    model, _, trans = construct_full_model(args.model.config)
    model.load_state_dict(torch.load(args.model.weight))
    model.to(args.base.device).eval()

    # Get all images folders
    results_rows_FIQA = []
    intensities = []
    images_folders = os.listdir(path_to_main_folder)
    for img_fold in images_folders:
        attributes_fold = os.listdir(path_to_main_folder + img_fold + "/")
        for attribute in attributes_fold:
            # Get all files in folder
            if os.path.isdir(path_to_main_folder + img_fold + "/" + attribute + "/"):
                files = os.listdir(path_to_main_folder + img_fold + "/" + attribute + "/")
                files.sort()

                # Construct the Image Dataloader
                dataset = InferenceDatasetWrapper(path_to_main_folder + img_fold + "/" + attribute + "/", trans)
                dataloader = DataLoader(dataset, **args_to_dict(args.dataloader.params, {}))

                # Predict quality scores
                quality_scores = {}
                for (name_batch, img_batch) in tqdm(dataloader,desc=" Inference ", disable=not args.base.verbose):

                    img_batch = img_batch.to(args.base.device)
                    _, preds = model(img_batch)
                    preds = preds.detach().squeeze().cpu().numpy()
                    quality_scores.update(dict(zip(name_batch, preds)))

                #add row
                quality_scores = dict(sorted(quality_scores.items()))
                temp_row = []
                temp_row.append(img_fold)
                temp_row.append(attribute)
                for k,v in quality_scores.items():
                    intensity = k.split("/")[-1].split("_")[-1].split(".png")[0]
                    if intensity == "original":continue
                    #print(round(float(v),4),type(round(float(v),4)),float(0.5694),type(float(0.5694)),float(0.5694)==round(float(v),4))
                    if round(float(v),4) == float(0.5694):
                        temp_row.append("None")
                    else:
                        temp_row.append(v)
                    if intensity not in intensities:
                        intensities.append(intensity)
                results_rows_FIQA.append(temp_row)


    # Save results
    base_row = ["Img_name", "Attribute"]
    for intensity in intensities:
            base_row.append(intensity)
    results_rows_FIQA.insert(0, base_row)

    with open(path_for_results_csv + results_csv_name_FIQA, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(results_rows_FIQA)


import statistics
def mean_var_sq(args,path_folder):

    # Seed all libraries to ensure consistency between runs.
    seed_all(args.base.seed)

    # Load the trained AI-KD model and construct the transformation
    model, _, trans = construct_full_model(args.model.config)
    model.load_state_dict(torch.load(args.model.weight))
    model.to(args.base.device).eval()

    # Get all images folders
    # Construct the Image Dataloader
    dataset = InferenceDatasetWrapper(path_folder, trans)
    dataloader = DataLoader(dataset, **args_to_dict(args.dataloader.params, {}))

    # Predict quality scores
    quality_scores = {}
    for (name_batch, img_batch) in tqdm(dataloader,desc=" Inference ", disable=not args.base.verbose):

        img_batch = img_batch.to(args.base.device)
        _, preds = model(img_batch)
        preds = preds.detach().squeeze().cpu().numpy()
        quality_scores.update(dict(zip(name_batch, preds)))

    #add row
    quality_scores = dict(sorted(quality_scores.items()))
    scores = list(quality_scores.values())
    print(scores)

    #Convert to float 64 because error with statistics var
    for s in range(len(scores)):
        scores[s]=float(scores[s])
    mean = statistics.mean(scores)
    var = statistics.variance(scores)
    std = statistics.stdev(scores)

    print("Mean ",mean)
    print("std ", std)
    print("var ", var)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",type=str,help=' Location of the AI-KD inference configuration. ')
    """parser.add_argument('--img_folder', help='root folder containing generated images')
    parser.add_argument('--csv_folder', help='path to folder containing csv on id eval')
    parser.add_argument('--csv_name_fiqa', help='csv name for this lpips')"""

    args = parser.parse_args()
    arguments = parse_config_file(args.config)

    # create if folder no exist
    """if not os.path.isdir(args.csv_folder):
        os.makedirs(args.csv_folder)

    inference(arguments,args.img_folder,args.csv_folder,args.csv_name_fiqa)"""

    path_folder = "/home/bour231/Desktop/icfegm/data/CelebA/aligned/"
    print("For ",path_folder)
    mean_var_sq(arguments, path_folder)