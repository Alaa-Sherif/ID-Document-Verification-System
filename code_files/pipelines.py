from code_files.pre_process import pre_process
from code_files.id.classification import id_classification
from code_files.id.ocr.id_ocr import IdOCR, CustomError
from code_files.graduation_certificate.classification import BC_Classifier
from code_files.military_certificate.classification import MC_Classification
from code_files.military_certificate.ocr import m_certificate_ocr
from code_files.personal_image.personal_image_validation import single_face_mesh

import cv2
import os
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt


################################################################ Start of ID Logic ################################################################
def id_pipeline(img, model):
    # apply image preprocessing
    img = pre_process.orient(img)
    img = pre_process.docTr_wrapper(img)

    # classify the image
    predicted_label = id_classification.classify_ID_img(img, model)

    # ---------------------------------------------------------------------------------------
    id_ocr = IdOCR()
    # check image for the next stage
    if len(predicted_label) == 1 and predicted_label[0] != 'ID':
        raise CustomError("Not_ID")
    
    # then move to the next stage
    try:
        name, pred_id = id_ocr.apply_ocr(img)

    except CustomError as e:
        raise CustomError(e.message)           
    
    else:
        return name, pred_id, img
################################################################ End of ID Logic ################################################################

################################################################ Start of Grad Certificate Logic ################################################################

def bachelor_cert_pipeline(img, model):
    img = pre_process.orient(img)
    img = pre_process.docTr_wrapper(img)

    img_label = BC_Classifier.BC_Classifier(img, model)
    if img_label == 'Other':
        # raise CustomError("Oops!! Wrong Image\nPlease upload a clear image for your Bachelor Certificate")
        raise CustomError("Not_Bachelor_Certificate")

    return img
    
################################################################ End of Grad Certificate Logic ################################################################

################################################################ Start of Military Certificate Logic ################################################################
def military_cert_pipeline(img, model):
    # img = cv2.imread(img_path)
    img = pre_process.orient(img)
    img = pre_process.docTr_wrapper(img)

    img_label = MC_Classification.MC_Classifier(img, model)

    if img_label != 'Other':
        # Move to the next stage
        military_result = m_certificate_ocr.prepro(img)
        
        if military_result == "not decided" :
            military_result = m_certificate_ocr.prepro(img,contrast=True)
    else:
        # raise CustomError("Oops!! Wrong Image\nPlease upload a clear image for your Military Certificate")
        raise CustomError("Not_Military_Certificate")

    return military_result, img
################################################################ End of Military Certificate Logic ################################################################

def image_pipeline(image, image_id):
    msg , image = single_face_mesh(image, image_id)
    if msg == 'Accepted':
        return image

    raise CustomError(msg)


################################################################### Test the full pipeline #############################

def full_pipeline(data):
    id_classifier = load_model('code_files/id/classification/CNN_ID_Classifier.h5')
    bc_Classifier = load_model("code_files/graduation_certificate/classification/CNN_BC_Classifier2.h5")
    mc_classifier = load_model('code_files/military_certificate/classification/CNN_MC_Classifier.h5')

    for i in range(1500, len(data.image_path)):
        image = data.image_path[i]
        img = cv2.imread(image)
        plt.imshow(img)
        plt.show()
        try:
            if 'Personal_Id' in image:
                person_name, person_id, id_img = id_pipeline(img, id_classifier)
                print("name:", person_name)
                print("ID:", person_id)
                plt.imshow(id_img)
                plt.show()
            
            elif 'Bachelor_Certificate' in image:
                bc_img = bachelor_cert_pipeline(img, bc_Classifier)
                plt.imshow(bc_img)
                plt.show()
            
            elif 'Military_Certificate' in image:
                res, mc_img = military_cert_pipeline(img, mc_classifier)
                print("Status:", res)
                plt.imshow(mc_img)
                plt.show()
        except:
            continue
        # else:
        #     my_img = image_pipeline(image, i)
        #     plt.imshow(my_img)
        #     plt.show()

    return "Done!\n"
           

# data = pd.read_csv("test_data.csv")
# # images_path = data['image_path']
# message = full_pipeline(data)
# print(message)


