import os
import csv
def remove_leading_zeros(num_str):
    num_str = num_str.split(".jpg")[0]
    return str(int(num_str))+".jpg"

#For CelebA
#For images in align, use name to only keep necessayr from .txt
with open('identity_CelebA.txt', 'r') as read_obj:
    csv_reader = csv.reader(read_obj)
    list_of_csv = list(csv_reader)
images_celebA = os.listdir("/home/bour231/Desktop/gan_inversion/data_test/CelebA_Hq/img")
new_id_celebA = []
for i in list_of_csv:
    if remove_leading_zeros(i[0].split(" ")[0]) in images_celebA:
        new_id_celebA.append([remove_leading_zeros(i[0].split(" ")[0]).split(".jpg")[0]+".png",i[0].split(" ")[1]])
print(new_id_celebA)
with open("id_CelebA_Hq.csv", 'w') as f:
    write = csv.writer(f)
    write.writerows(new_id_celebA)
    
    
    
#For Feret
#Create txt using images names, names + counter, add to counter only if image t != image t-1 names
images_feret = os.listdir("/home/bour231/Desktop/gan_inversion/data_test/Feret/img")
id_Feret = []
for img in images_feret:
    id_Feret.append([img,img.split("_")[0]])
print(id_Feret)
with open("id_Feret.csv", 'w') as f:
    write = csv.writer(f)
    write.writerows(id_Feret)
    
    
    
#For FRL
#one image per identity, just do image name + counter
images_frl = os.listdir("/home/bour231/Desktop/gan_inversion/data_test/FRL/img")
id_FRL = []
counter = 0
for img in images_frl:
    id_FRL.append([img.split(".jpg")[0]+".png",counter])
    counter += 1
print(id_FRL)
with open("id_FRL.csv", 'w') as f:
    write = csv.writer(f)
    write.writerows(id_FRL)
