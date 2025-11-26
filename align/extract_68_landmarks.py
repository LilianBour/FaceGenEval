import dlib
import cv2
import os
import csv

def extract_landmarks(image,frontalFaceDetector,faceLandmarkDetector):
    img = cv2.imread(image)
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    allFaces = frontalFaceDetector(imageRGB, 0)

    # Get Landmarks for each face in the image
    for k in range(0, len(allFaces)):
        faceRectangle = dlib.rectangle(int(allFaces[k].left()), int(allFaces[k].top()), int(allFaces[k].right()),int(allFaces[k].bottom()))
        all_Landmarks_detected = faceLandmarkDetector(imageRGB, faceRectangle)

        landmarks_list = []
        # add image name to row
        landmarks_list.append(image.split("/")[-1])
        for p in all_Landmarks_detected.parts():
            landmarks_list.append(int(p.x))
            landmarks_list.append(int(p.y))

    try:
        return landmarks_list
    except:
        return []

if __name__ == '__main__':
    #Load Model
    Model_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/models/" + "shape_predictor_68_face_landmarks.dat"
    frontalFaceDetector = dlib.get_frontal_face_detector()
    faceLandmarkDetector = dlib.shape_predictor(Model_PATH)

    root_folder = "./data/data_test/"
    list_images = os.listdir(root_folder + "images/")

    list_landmarks = []
    for img in list_images:
        landmarks_for_img = extract_landmarks(root_folder + "images/" + img, frontalFaceDetector, faceLandmarkDetector)
        list_landmarks.append(landmarks_for_img)

    with open(root_folder + "landmarks.txt", "w") as f:
        wr = csv.writer(f,delimiter=" ")
        wr.writerows(list_landmarks)
