import streamlit as st

from pathlib import Path
from tensorflow import keras
from keras.models import load_model
import cv2

from code_files.pre_process import pre_process
from code_files.id.classification import id_classification
from code_files.id.ocr.id_ocr import IdOCR, CustomError
from code_files.graduation_certificate.classification import BC_Classifier
from code_files.graduation_certificate.ocr import g_certificate_ocr
from code_files.military_certificate.classification import MC_Classification
from code_files.military_certificate.ocr import m_certificate_ocr
from code_files.personal_image.personal_image_validation import single_face_mesh

################################################################ Start of Main Page Dash Code ################################################################
def main():
    # Set page title
    st.title("ID & Documents Verification System")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Home", "Upload Documents"))

    uploaded_images = []  # New list to store uploaded images

    if page == "Home":
        show_home()

    elif page == "Upload Documents":
        uploaded_images = show_upload_documents()

def show_home():
    st.header("Home Page")
    st.image(".\\assets\\Technologies-section-–-ID-reader-300x204.png", use_column_width=True)

def show_upload_documents():
    st.header("Upload Documents")
    gender = st.radio("Select your gender", ("Male", "Female"))

    uploaded_images = []  # New list to store uploaded images

    if gender == "Male":
        uploaded_images = upload_documents(4, "Personal ID", "Personal img", "Bachelor Cert", "Military Cert")
    elif gender == "Female":
        uploaded_images = upload_documents(3, "Personal ID", "Personal Img", "Bachelor Cert")

    return uploaded_images  # Return the uploaded images list

def upload_documents(num_boxes, *section_names):
    columns = st.columns(num_boxes)

    images = [
        ".\\assets\\id.png",
        ".\\assets\\focus.png",
        ".\\assets\\bachelor.png",
        ".\\assets\\military.png"
    ]

    uploaded_images = []  # New list to store uploaded images

    for i in range(num_boxes):
        with columns[i]:
            image_path = images[i]
            st.image(image_path, caption=section_names[i], use_column_width=True)
            uploaded_file = st.file_uploader("Upload a photo", type="jpg", key=f"image_{i}")

            if uploaded_file is not None:
                # Perform image validation
                validation_result = validate_image(uploaded_file, i)
                if validation_result:
                    st.success("Validation: Image is valid")
                    uploaded_images.append(uploaded_file)  # Append the uploaded image to the list
                else:
                    st.error("Validation: Image is invalid")

    return uploaded_images  # Return the uploaded images list
################################################################ End of Main Page Dash Code ################################################################


################################################################ Start of ID Logic ################################################################
def id_pipeline(img_path, model):
    # read the image
    img = cv2.imread(img_path)

    # apply image preprocessing
    img = pre_process.orient(img)

    # classify the image
    predicted_label = id_classification.classify_ID_img(img, model)

    # ---------------------------------------------------------------------------------------
    id_ocr = IdOCR()
    # check image for the next stage
    if len(predicted_label) == 1 and predicted_label[0] != 'ID':
        raise CustomError("Oops!! Wrong Image\nPlease upload a clear image for your National ID")
    
    # then move to the next stage
    try:
        name, pred_id = id_ocr.apply_ocr(img)

    except CustomError as e:
        raise CustomError(e.message)           
    
    else:
        return name, pred_id
################################################################ End of ID Logic ################################################################


################################################################ Start of Grad Certificate Logic ################################################################

def bachelor_cert_pipeline(img_path, model):
    img = cv2.imread(img_path)

    img_label = BC_Classifier.BC_Classifier(img, model)
    if img_label == 'Other':
        print("Oops!! Wrong Image\nPlease upload a clear image for your Bachelor Certificate")
        return False
    
################################################################ End of Grad Certificate Logic ################################################################

################################################################ Start of Military Certificate Logic ################################################################
def military_cert_pipeline(img_path, model):
    img = cv2.imread(img_path)
    img = pre_process.orient(img)
    img_label = MC_Classification.MC_Classifier(img, model)

    if img_label != 'Other':
        # Move to the next stage
        message = m_certificate_ocr.prepro(img_path)
        
        if message == "not decided" :
            message = m_certificate_ocr.prepro(img_path,contrast=True)
    else:
        message = "Oops!! Wrong Image\nPlease upload a clear image for your Military Certificate"
        return False

    return message   
################################################################ End of Military Certificate Logic ################################################################

def validate_image(File, i):
    # 0: Personal ID
    # 1: Personal Img
    # 2: Bachelor Cert
    # 3: Military Cert
    save_folder = '.\\imgs'
    save_path = Path(save_folder, File.name)
    with open(save_path, mode='wb') as w:
        w.write(File.getvalue())

    if save_path.exists():
        st.success(f'File {File.name} is successfully saved!')

        if i == 0:
            id_classifier = load_model('code_files/id/classification/CNN_ID_Classifier.h5')
            try:
                res = id_pipeline('.\\imgs\\'+File.name, id_classifier)
                st.success(f'Name: {res[0]}, ID: {res[1]}')
                return True # id, name
            except CustomError as e:
                st.error(e.message)
                return False
        
        if i == 1:
            # img = pre_process.orient(img)
            msg , _ = single_face_mesh('.\\imgs\\'+File.name, save_folder)
            if msg == 'Accepted':
                return True
            st.error(f'{msg}')
            return False # maybe return msg
        
        if i == 2:
            bc_Classifier = load_model("code_files/graduation_certificate/classification/CNN_BC_Classifier2.h5")

            results = bachelor_cert_pipeline('.\\imgs\\'+File.name, bc_Classifier)
            if results:
                st.success(f'{results}')
                return True
            return False 
        
        if i == 3:
            mc_classifier = load_model('code_files/military_certificate/classification/CNN_MC_Classifier.h5')

            mc_results = military_cert_pipeline('.\\imgs\\'+File.name, mc_classifier)
            if mc_results:
                st.success(f'{mc_results}')
                return True
            return False

    return True

if __name__ == "__main__":
    main()
