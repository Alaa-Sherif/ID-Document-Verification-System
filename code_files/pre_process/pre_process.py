from check_orientation.pre_trained_models import create_model
import pandas as pd
import numpy as np
import cv2
import torch
import albumentations as albu
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
import sys
import os
import shutil
import math
from PIL import Image
# from docTr_updated import inference


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

distorrted_path = os.path.join('.', 'code_files', 'pre_process', 'docTr_updated', 'distorted') + '\\'
gsave_path = os.path.join('.', 'code_files', 'pre_process', 'docTr_updated', 'geo_rec') + '\\'
isave_path = os.path.join('.', 'code_files', 'pre_process', 'docTr_updated', 'ill_rec') + '\\'
Seg_path = os.path.join('.', 'code_files', 'pre_process', 'docTr_updated', 'model_pretrained', 'seg.pth')
GeoTr_path = os.path.join('.', 'code_files', 'pre_process', 'docTr_updated', 'model_pretrained', 'geotr.pth')
IllTr_path = os.path.join('.', 'code_files', 'pre_process', 'docTr_updated', 'model_pretrained', 'illtr.pth')
ill_rec = False

args = Namespace(
    distorrted_path=distorrted_path,
    gsave_path=gsave_path,
    isave_path=isave_path,
    Seg_path=Seg_path,
    GeoTr_path=GeoTr_path,
    IllTr_path=IllTr_path,
    ill_rec=ill_rec
)


model = create_model("swsl_resnext50_32x4d")
model.eval()
transform = albu.Compose([albu.Resize(height=224, width=224), albu.Normalize(p=1)], p=1)



opencv_home = cv2.__file__
folders = opencv_home.split(os.path.sep)[0:-1]
path = folders[0]
for folder in folders[1:]:
	path = path + "\\" + folder
path_for_face = path+"\\data\\haarcascade_frontalface_default.xml"
path_for_eyes = path+"\\data\\haarcascade_eye.xml"
path_for_nose = path+"\\data\\haarcascade_mcs_nose.xml"

if os.path.isfile(path_for_face) != True:
	raise ValueError(
		"opencv is not installed pls install using pip install opencv ",
	path_for_face, " violated.")

face_detector = cv2.CascadeClassifier(path_for_face)
eye_detector = cv2.CascadeClassifier(path_for_eyes)
nose_detector = cv2.CascadeClassifier(path_for_nose)



def test():
    return 'Hello World!'


def orient(img):
    """This function takes an image and returns the same one but oriented correctly"""
    temp = []
    for k in [0]:
        x = transform(image=np.rot90(img, k))["image"]
        temp += [tensor_from_rgb_image(x)]
    with torch.no_grad():
        prediction = model(torch.stack(temp)).numpy()
    preds = [str(round(tx, 2)) for tx in prediction[0]]
    k = preds.index(max(preds))
    return np.rot90(img, -k)


def adaptive_threasholding(img):
    """This is binarizing that performs better than normal threasholding. Takes an image and returns another one"""
    # smoothing first
    blurred = cv2.GaussianBlur(img,(5,5),0)
    
    # Using adaptive Thresholding
    im_gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(im_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,11,2)
    return thresh


def docTr_wrapper(img, illumination=False):
    try:
        from code_files.pre_process.docTr_updated import inference
        # sys.path.append(os.path.join('.', 'pre_process', 'docTr_updated'))
        # import inference
        
    except Exception as e:
        print('the docTr path is not here, please make it nearby')
        print(e)
        return img
    else:
        cv2.imwrite(os.path.join('.', 'code_files', 'pre_process', 'docTr_updated', 'distorted', 'img.png'), img[:, :, ::-1])
        if illumination:
            args.ill_rec = True
            inference.rec(args)
            return cv2.imread(os.path.join('.', 'code_files', 'pre_process', 'docTr_updated', 'ill_rec', 'img_ill.png'))[:, :, ::-1]
        args.ill_rec = False
        inference.rec(args)
        return cv2.imread(os.path.join('.', 'code_files', 'pre_process', 'docTr_updated', 'geo_rec', 'img_geo.png'))[:, :, ::-1]





# Detect face
def face_detection(img):
	faces = face_detector.detectMultiScale(img, 1.1, 4)
	if (len(faces) <= 0):
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		return img, img_gray
	else:
		X, Y, W, H = faces[0]
		img = img[int(Y):int(Y+H), int(X):int(X+W)]
		return img, cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)


def trignometry_for_distance(a, b):
	return math.sqrt(((b[0] - a[0]) * (b[0] - a[0])) +\
					((b[1] - a[1]) * (b[1] - a[1])))

# Find eyes
def face_alignment(img_raw):
    img, gray_img = face_detection(img_raw.copy())
    eyes = eye_detector.detectMultiScale(gray_img)

    # for multiple people in an image find the largest
    # pair of eyes
    if len(eyes) >= 2:
        eye = eyes[:, 2]
        container1 = []
        for i in range(0, len(eye)):
            container = (eye[i], i)
            container1.append(container)
        df = pd.DataFrame(container1, columns=[
                        "length", "idx"]).sort_values(by=['length'])
        eyes = eyes[df.idx.values[0:2]]

        # deciding to choose left and right eye
        eye_1 = eyes[0]
        eye_2 = eyes[1]
        if eye_1[0] > eye_2[0]:
            left_eye = eye_2
            right_eye = eye_1
        else:
            left_eye = eye_1
            right_eye = eye_2

        # center of eyes
        # center of right eye
        right_eye_center = (
            int(right_eye[0] + (right_eye[2]/2)),
        int(right_eye[1] + (right_eye[3]/2)))
        right_eye_x = right_eye_center[0]
        right_eye_y = right_eye_center[1]
        cv2.circle(img, right_eye_center, 2, (255, 0, 0), 3)

        # center of left eye
        left_eye_center = (
            int(left_eye[0] + (left_eye[2] / 2)),
        int(left_eye[1] + (left_eye[3] / 2)))
        left_eye_x = left_eye_center[0]
        left_eye_y = left_eye_center[1]
        cv2.circle(img, left_eye_center, 2, (255, 0, 0), 3)

        # finding rotation direction
        if left_eye_y > right_eye_y:
            print("Rotate image to clock direction")
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1  # rotate image direction to clock
        else:
            print("Rotate to inverse clock direction")
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1  # rotate inverse direction of clock


        # calculating the angle between a, b, c
        cv2.circle(img, point_3rd, 2, (255, 0, 0), 2)
        a = trignometry_for_distance(left_eye_center, point_3rd)
        b = trignometry_for_distance(right_eye_center, point_3rd)
        c = trignometry_for_distance(right_eye_center, left_eye_center)
        cos_a = (b*b + c*c - a*a)/(2*b*c)
        angle = (np.arccos(cos_a) * 180) / math.pi

        if direction == -1:
            angle = 90 - angle
        # else:
        #     angle = -(90-angle)

        # rotate image
        new_img = Image.fromarray(img_raw)
        new_img = np.array(new_img.rotate(direction * angle))

        return new_img
    return img


def align_face(img, zoomed=False):
    alignedFace = face_alignment(img)
    if zoomed:
        img, _ = face_detection(alignedFace)
        return img
    return alignedFace