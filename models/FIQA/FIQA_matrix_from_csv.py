import csv
import os
import argparse
from matplotlib import pyplot as plt
import numpy as np

def plot_matrix(cm, cms,  labels, intensities,model,cmap=plt.cm.Greens):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(intensities))
    plt.xticks(tick_marks, intensities, rotation=45)
    tick_marks = np.arange(len(labels))
    plt.yticks(tick_marks, labels)

    thresh = np.mean(cm)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, '{0:.2f}'.format(cm[i, j]) + '\n$\pm$' + '{0:.2f}'.format(cms[i, j]),horizontalalignment="center",verticalalignment="center", fontsize=7,color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Attributes')
    plt.xlabel('Intensity')
    plt.title("FIQA for "+model)

def unique(list1):
    list_set = set(list1)
    unique_list = (list(list_set))
    return unique_list
def extract_fiqa(path_csv):
    with open(path_csv, newline='') as f:
        reader = csv.reader(f)
        csv_fiqa = list(reader)

    #Get attributes and intensities
    attribute_list = []
    baseline_index = 2
    for row_i in csv_fiqa[1:]:
        attribute_list.append(row_i[1])
    attribute_list = unique(attribute_list)

    #Create dict with empty list for each intensity
    dict_attribute_fiqa = {}
    for att in attribute_list:
        dict_attribute_fiqa[att]=[]

    dict_attribute_fiqa = dict(sorted(dict_attribute_fiqa.items()))

    #Add fiqa values in dict
    for row_i in csv_fiqa[1:]:
        to_append_as_float = []
        for i in row_i[baseline_index:]:
            if i != "None":
                to_append_as_float.append(float(i))
            else:
                to_append_as_float.append(np.nan)
        dict_attribute_fiqa[row_i[1]].append(to_append_as_float)

    return dict_attribute_fiqa


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quality_folder', help='root folder containing quality fiqa csv')
    args = parser.parse_args()
    path_a = args.quality_folder

    models_intensity_list = {"vecgan": ["-0.50","0.00","0.50"], "diffae": ["-0.3",0.0,0.3], "stargan":[1.00]} #string to force two decimals for some values

    models_ = os.listdir(path_a)
    models = []
    for m in models_:
        if ".png" not in m:
            models.append(m)

    for model in models:
        csv_name = path_a + model + "/" + "fiqa_quality_"+model+".csv"
        dict_attribute_fiqa = extract_fiqa(csv_name)
        dict_attribute_fiqa_sd = dict_attribute_fiqa.copy()
        intensity_list = models_intensity_list[model]

        #Compute mean for each attribute
        labels = []
        fiqa_means = []
        fiqa_std = []
        for att,fiqa in dict_attribute_fiqa.items():
            fiqa_means.append(np.nanmean(dict_attribute_fiqa[att], axis=0))
            fiqa_std.append(np.nanstd(dict_attribute_fiqa_sd[att], axis=0))
            labels.append(att)

            dict_attribute_fiqa[att] = list(np.mean(dict_attribute_fiqa[att],axis=0))
            dict_attribute_fiqa_sd[att] = list(np.std(dict_attribute_fiqa_sd[att],axis=0))

        fiqa_means = np.array(fiqa_means)
        fiqa_std = np.array(fiqa_std)

        plt.figure()
        plot_matrix(fiqa_means, fiqa_std, labels,intensity_list,model)
        save_name = path_a +"fiqa_"+model+".png"
        plt.savefig(save_name, format="png", bbox_inches="tight")

