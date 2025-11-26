import argparse

import cv2
import numpy as np
import torch
import os

from models.swinface_project.model_inference import build_model

@torch.no_grad()

def load_model(cfg, weight):
    model = build_model(cfg)

    dict_checkpoint = torch.load(weight)

    model.backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
    model.fam.load_state_dict(dict_checkpoint["state_dict_fam"])
    model.tss.load_state_dict(dict_checkpoint["state_dict_tss"])
    model.om.load_state_dict(dict_checkpoint["state_dict_om"])

    model.eval()

    return model
def inference(model, img):

    img = cv2.cvtColor((np.array(img)),cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    output = model(img)#.numpy()

    return output

def get_binary_classification_(output):
    with torch.no_grad():
        attribute_list_output = []
        binary_attribute_output = []
        binary_attribute_output_raw = []

        for result in output:
            if result not in ["Age", "Expression", "Recognition"]:
                # Correct labels to match celebA
                if result == "Five O'Clock Shadow":
                    attribute_list_output.append("5_o_Clock_Shadow")
                elif result == "Gender":
                    attribute_list_output.append("Male")
                else:
                    attribute_list_output.append(result)
                binary_attribute_output_raw.append([output[result][0].numpy()[0], output[result][0].numpy()[1]])

                #IF attribute have high value (superior to 1 or inferior to -1) the attribute value is negative and thus not present on the face
                if output[result][0].numpy()[0] > 1 and output[result][0].numpy()[1] < -1:
                    binary_attribute_output.append(-1)
                else:
                    binary_attribute_output.append(1)

        #Sort to match CelebA
        attribute_list_output, binary_attribute_output, binary_attribute_output_raw = zip(*sorted(zip(attribute_list_output, binary_attribute_output, binary_attribute_output_raw)))
        sorted_binary_attribute_output = list(binary_attribute_output)

        return sorted_binary_attribute_output

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def get_binary_classification__(output):
    with torch.no_grad():
        attribute_list_output = []
        binary_attribute_output = []
        binary_attribute_output_raw = []

        for result in output:
            if result not in ["Age", "Expression", "Recognition"]:
                # Correct labels to match celebA
                if result == "Five O'Clock Shadow":
                    attribute_list_output.append("5_o_Clock_Shadow")
                elif result == "Gender":
                    attribute_list_output.append("Male")
                else:
                    attribute_list_output.append(result)
                binary_attribute_output_raw.append([output[result][0].numpy()[0], output[result][0].numpy()[1]])

                #IF attribute have high value (superior to 1 or inferior to -1) the attribute value is negative and thus not present on the face
                #Test with softmax output
                softmaxed = softmax(output[result].numpy()[0])#Position 0 is for NO, 1 for Yes
                if softmaxed[0] > softmaxed[1]: #No > Yes
                    binary_attribute_output.append(-1)
                else:
                    binary_attribute_output.append(1)

        #Sort to match CelebA
        attribute_list_output, binary_attribute_output, binary_attribute_output_raw = zip(*sorted(zip(attribute_list_output, binary_attribute_output, binary_attribute_output_raw)))
        sorted_binary_attribute_output = list(binary_attribute_output)

        return sorted_binary_attribute_output


def get_binary_classification(output):
    """
    Only for 10 attributes : "Bangs", "Black_hair", "Brown_hair", "Blond_hair", "Eyeglasses", "Heavy_Makeup", "Male", "No_Beard", "Smiling", "Young"
    """
    attr_to_keep = ["Bangs", "Black Hair", "Brown Hair", "Blond Hair", "Eyeglasses", "Heavy Makeup", "Male", "No Beard", "Smiling", "Young"]
    with torch.no_grad():
        attribute_list_output = []
        binary_attribute_output = []
        binary_attribute_output_raw = []

        for result in output:
            if result not in ["Age", "Expression", "Recognition"]:
                # Correct labels to match celebA
                if result == "Five O'Clock Shadow":
                    attribute_list_output.append("5_o_Clock_Shadow")
                elif result == "Gender":
                    attribute_list_output.append("Male")
                else:
                    attribute_list_output.append(result)
                binary_attribute_output_raw.append([output[result][0].numpy()[0], output[result][0].numpy()[1]])

                #IF attribute have high value (superior to 1 or inferior to -1) the attribute value is negative and thus not present on the face
                #Test with softmax output
                softmaxed = softmax(output[result].numpy()[0])#Position 0 is for NO, 1 for Yes
                if softmaxed[0] > softmaxed[1]: #No > Yes
                    binary_attribute_output.append(-1)
                else:
                    binary_attribute_output.append(1)

        #Sort to match CelebA
        attribute_list_output, binary_attribute_output, binary_attribute_output_raw = zip(*sorted(zip(attribute_list_output, binary_attribute_output, binary_attribute_output_raw)))

        light_attr_list = []
        attribute_list_output = list(attribute_list_output)

        binary_attribute_output = list(binary_attribute_output)
        for att_i in range(len(attribute_list_output)):
            #if attribute_list_output[att_i] in attr_to_keep:
            light_attr_list.append(binary_attribute_output[att_i])
        return light_attr_list

def get_attributes(output):
    """
    Only for 10 attributes : "Bangs", "Black_hair", "Brown_hair", "Blond_hair", "Eyeglasses", "Heavy_Makeup", "Male", "No_Beard", "Smiling", "Young"
    """
    attr_to_keep = ["Bangs", "Black Hair", "Brown Hair", "Blond Hair", "Eyeglasses", "Heavy Makeup", "Male", "No Beard", "Smiling", "Young"]
    with torch.no_grad():
        attribute_list_output = []
        binary_attribute_output = []
        binary_attribute_output_raw = []

        for result in output:
            if result not in ["Age", "Expression", "Recognition"]:
                # Correct labels to match celebA
                if result == "Five O'Clock Shadow":
                    attribute_list_output.append("5_o_Clock_Shadow")
                elif result == "Gender":
                    attribute_list_output.append("Male")
                else:
                    attribute_list_output.append(result)
                binary_attribute_output_raw.append([output[result][0].numpy()[0], output[result][0].numpy()[1]])

                #IF attribute have high value (superior to 1 or inferior to -1) the attribute value is negative and thus not present on the face
                #Test with softmax output
                softmaxed = softmax(output[result].numpy()[0])#Position 0 is for NO, 1 for Yes
                if softmaxed[0] > softmaxed[1]: #No > Yes
                    binary_attribute_output.append(-1)
                else:
                    binary_attribute_output.append(1)

        #Sort to match CelebA
        attribute_list_output, binary_attribute_output, binary_attribute_output_raw = zip(*sorted(zip(attribute_list_output, binary_attribute_output, binary_attribute_output_raw)))
        light_attr_list = []
        attribute_list_output = list(attribute_list_output)
        for att_i in range(len(attribute_list_output)):
            #if attribute_list_output[att_i] in attr_to_keep:
            light_attr_list.append(attribute_list_output[att_i])
        return light_attr_list
class SwinFaceCfg:
    network = "swin_t"
    fam_kernel_size=3
    fam_in_chans=2112
    fam_conv_shared=False
    fam_conv_mode="split"
    fam_channel_attention="CBAM"
    fam_spatial_attention=None
    fam_pooling="max"
    fam_la_num_list=[2 for j in range(11)]
    fam_feature="all"
    fam = "3x3_2112_F_s_C_N_max"
    embedding_size = 512


correct_attribute_order = ["5_o_Clock_Shadow",
                           "Arched_Eyebrows", "Attractive",
                           "Bags_Under_Eyes",
                           "Bald",
                           "Bangs",
                           "Big_Lips",
                           "Big_Nose",
                           "Black_Hair",
                           "Blond_Hair",
                           "Blurry",
                           "Brown_Hair",
                           "Bushy_Eyebrows",
                           "Chubby",
                           "Double_Chin",
                           "Eyeglasses",
                           "Goatee",
                           "Gray_Hair",
                           "Heavy_Makeup",
                           "High_Cheekbones",
                           "Male",
                           "Mouth_Slightly_Open",
                           "Mustache",
                           "Narrow_Eyes",
                           "No_Beard",
                           "Oval_Face",
                           "Pale_Skin",
                           "Pointy_Nose",
                           "Receding_Hairline",
                           "Rosy_Cheeks",
                           "Sideburns",
                           "Smiling",
                           "Straight_Hair",
                           "Wavy_Hair",
                           "Wearing_Earrings",
                           "Wearing_Hat",
                           "Wearing_Lipstick",
                           "Wearing_Necklace",
                           "Wearing_Necktie",
                           "Young"]

def extrac_classif_folder(path_to_folder,model):
    list_images = os.listdir(path_to_folder)

    results = []
    results.append([len(list_images)])
    results.append(correct_attribute_order)

    counter = 0
    for img in list_images :
        counter+=1
        output = inference(model, path_to_folder + img)
        results_classif_img = get_binary_classification(output)
        results_classif_img.insert(0, img)
        results.append(results_classif_img)

    return results

def compare_two_faces(img_1,img_2,model):
    output_1 = inference(model, img_1)
    output_2 = inference(model, img_2)

    with torch.no_grad():
        emb_1 = output_1["Recognition"][0].numpy()
        emb_2 = output_2["Recognition"][0].numpy()

    # normalize vectors with l2
    emb_1_l2 = emb_1 / np.linalg.norm(emb_1)
    emb_2_l2 = emb_2 / np.linalg.norm(emb_2)

    # Cosine similarity
    cos_sim = np.dot(emb_1_l2, emb_2_l2)
    cos_distance = 1 - cos_sim

    return cos_distance

if __name__ == "__main__":
    cfg = SwinFaceCfg()

    model_path = "/home/bour231/Desktop/Swinface/Model/checkpoint_step_79999_gpu_0.pt"
    image_path = "/home/bour231/Desktop/Swinface/Data/1279.jpg"

    model = load_model(cfg, model_path)
    output = inference(model, image_path)

    with torch.no_grad():
        for each in output:
            print(each,output[each][0].numpy())





