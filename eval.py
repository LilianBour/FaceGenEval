import argparse
import csv
import os
import sys
import yaml

import numpy as np

from datetime import datetime
from PIL import Image

from models.swinface_project.inference import inference, get_attributes
from metrics.fic import embedding_list_for_data
from metrics.fac import fac_img_i, fac
from metrics.metrics import compute_metrics, average_metrics, std_metrics
from utils.utils import save_combined_images_and_individual_metrics, load_models

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = Config(value)  # recursive conversion

    __getattr__ = dict.get

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    opts = Config(yaml.safe_load(open(args.config_file)))
    size_of_images = opts.options.image_size
    main_path_data = opts.dir.exp_dir

    # Load models for evaluation
    models = load_models(opts)

    results_csv = []
    results_attributes = []
    dataset_names = opts.options.datasets

    #Evaluation for each dataset
    for dataset_name in dataset_names:
        print("###############################################"+dataset_name+"###############################################")
        # Replace path images by aligned images
        main_path_img = main_path_data + dataset_name + "/"
        path_images = main_path_img + "aligned"
        image_paths = [os.path.join(path_images, f) for f in os.listdir(path_images) if f.endswith(".png")]

        # get embeddings for all orignal images used in FIC
        if opts.metrics.fic_swinface:
            swinface_embds = embedding_list_for_data(image_paths, models["swinface"], "swinface")
        else: swinface_embds = None
        if opts.metrics.fic_arcface:
            arcface_embds = embedding_list_for_data(image_paths, models["arcface"], "arcface")
        else: arcface_embds = None

        #Loading identities of images for FIC
        if opts.options.multiple_id_datasets:
            with open(opts.options.multiple_id_datasets + 'id_' + dataset_name + '.csv', 'r') as read_obj:
                csv_reader = csv.reader(read_obj)
                list_name_id = list(csv_reader)
            data_ids = {a: b for a, b in list_name_id}
        else:
            #only one image per identity, FIC will be computer using an intra distance only between original image and reconstructed image and won't look for the closest with a similar identity
            data_ids = {}
            for img_i in range(len(image_paths)):
                data_ids[image_paths[img_i].split("/")[-1]]=img_i

        metric_names = [k.upper().replace("_", " ") for k, v in opts.metrics.items() if v]


        results_csv.append(["","","",dataset_name])
        results_attributes.append(["","","",dataset_name])
        results_csv.append(["variant"]+metric_names)

        #for each experiment setting
        variant_names = os.listdir(main_path_img + "/" + opts.options.experiment_folder_name + "/")
        for variant_name in variant_names:
            print("Doing",dataset_name, variant_name,"//time:",datetime.now().hour,datetime.now().minute)
            main_path_img_variant = main_path_img + "/" + opts.options.experiment_folder_name + "/" + variant_name + "/"

            #loop used if multiple attributes in experiment directory
            sub_experiments = os.listdir(main_path_img_variant)

            for sub_experiment in sub_experiments:
                metrics = {name: [] for name in metric_names}
                list_changes_classif = []
                path_outputs = main_path_img_variant + sub_experiment +"/"
                if not os.path.exists(path_outputs): os.makedirs(path_outputs)
                path_combined = opts.dir.output_dir + "combined/" + dataset_name + "/" +variant_name + "/"
                if not os.path.exists(path_combined): os.makedirs(path_combined)

                aligned_imgs = os.listdir(path_images)
                outputs_imgs = os.listdir(path_outputs)


                #debug option to try with few images
                quick_test = getattr(getattr(opts, "debug", {}), "quick_test", 0)
                if quick_test:
                    # instead of doing over the whole dataset, do on n images
                    image_paths = image_paths[0:opts.debug.quick_test]
                    aligned_imgs = aligned_imgs[0:opts.debug.quick_test]
                    outputs_imgs = outputs_imgs[0:opts.debug.quick_test]

                #Check if all image have been processed (edition or inversion)
                assert len(aligned_imgs) == len(outputs_imgs), "Not the same numbers of aligned images and reconstructed images"

                #Evaluation per output image
                for img_i in range(len(aligned_imgs)):
                    original_img = Image.open(path_images+"/"+aligned_imgs[img_i]).convert("RGB")
                    output_img = Image.open(path_outputs+"/"+aligned_imgs[img_i]).convert("RGB")
                    output_id = data_ids[aligned_imgs[img_i]] #identifier for image's identity


                    #add header attribute name in results concerning facial attribute if they're not
                    if len(results_attributes) <2 and opts.metrics.fac:
                        attributes = get_attributes(inference(models["swinface"], original_img))
                        results_attributes.append(["variant_name"]+attributes)

                    if opts.metrics.fac:
                        list_changes_classif.append(fac_img_i(models["swinface"],original_img, output_img))
                    metrics = compute_metrics(opts,metrics, original_img, output_img, output_id, models, data_ids, swinface_embds,arcface_embds)

                    if opts.options.save_merged_og_output:
                        combined_og_output = np.concatenate([np.array(original_img.resize((size_of_images, size_of_images))),np.array(output_img.resize((size_of_images, size_of_images)))], axis=1)
                        save_combined_images_and_individual_metrics(combined_og_output, metrics, path_combined,aligned_imgs[img_i])

                metrics_average = average_metrics(metrics)
                metrics_average.insert(0,variant_name+"_AVERAGE")
                if opts.options.std:
                    metrics_std = std_metrics(metrics)
                    metrics_std.insert(0, variant_name+"_STD")

                if opts.metrics.fac:
                    FIC_avg, FIC_std, sums_attribute_changes = fac(variant_name, list_changes_classif, len(aligned_imgs))
                    list_changes_classif.insert(len(list_changes_classif), [''])
                    list_changes_classif.insert(len(list_changes_classif), sums_attribute_changes)
                    results_attributes.append(sums_attribute_changes)
                    print("AVERAGE FAC", FIC_avg)
                    results_csv.append(metrics_average + [FIC_avg])
                    if opts.options.std:
                        results_csv.append(metrics_std + [FIC_std])

                if not os.path.exists(opts.dir.output_dir): os.makedirs(opts.dir.output_dir)
                #Save results
                with open(opts.dir.output_dir + "evaluation_results.csv", 'w') as f:
                    write = csv.writer(f)
                    write.writerows(results_csv)

                with open(opts.dir.output_dir + "evaluation_results_attributes.csv", 'w') as f:
                    write = csv.writer(f)
                    write.writerows(results_attributes)


                #Save full metrics
                path_full_metrics = opts.dir.output_dir + "/full_metrics/"+dataset_name+"/"
                if not os.path.exists(path_full_metrics): os.makedirs(path_full_metrics)

                #Save full metrics : attributes
                path_full_metrics_attr = path_full_metrics + "/attributes"+"/"
                if not os.path.exists(path_full_metrics_attr): os.makedirs(path_full_metrics_attr)
                with open(path_full_metrics_attr + variant_name +"_attributes.csv", 'w') as f:
                    write = csv.writer(f)
                    write.writerows(list_changes_classif)

                #Save full metrics : others
                nested_list = list(metrics.values())
                nested_list.insert(0,aligned_imgs)

                # Convert to numpy array to transpose
                nested_array = np.array(nested_list)
                nested_array = nested_array.T

                if not os.path.exists(path_full_metrics + "/full/" ): os.makedirs(path_full_metrics + "/full/" )
                with open(path_full_metrics + "/full/" + variant_name +"_metrics.csv", 'w') as f:
                    write = csv.writer(f)
                    write.writerows(nested_array)


