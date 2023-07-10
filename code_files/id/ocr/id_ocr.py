from ultralytics import YOLO
import pandas as pd
import os
import cv2
from easyocr import Reader
import numpy as np

TMP_PATH = 'output/id/tmp/'

################################################################ Start of Custom Error ################################################################
class CustomError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
################################################################ End of Custom Error ################################################################

################################################################ Start of Text Localizaer ################################################################
class TextLocalizer():
    model = YOLO("code_files/id/ocr/best_localize.pt")
    IMG_SIZE = 640
    img_path = TMP_PATH + 'tmp_img.jpg'
    classes = [0, 1 , 2]
    classes_names = {0: 'First Name', 1: 'Last Name', 2: 'ID Number'}


    def __init__(self) -> None:
        if not os.path.exists(TMP_PATH):
            os.makedirs(TMP_PATH)
        
    def preprocess_image(self, img):
        """
        Resizes an image to the specified size using OpenCV to pass it to yolo.

        Args:
            img: A NumPy array representing the image to be resized.

        Returns:
            A NumPy array representing the resized image.

        """

        return cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE), interpolation=cv2.INTER_AREA)
    
    def filter_bboxes(self, l_results):
        """
        Returns the most confident bounding boxes for each class in the input array in case any class is detected more than once.

        Args:
            self: An instance of the class.
            l_results (numpy.ndarray): A 2D array of bounding boxes and class labels.

        Returns:
            numpy.ndarray: A 1D array of the most confident bounding boxes for each class.
        """

        sorted_bbx = sorted(l_results.tolist(), key=lambda x: x[-1])
        sorted_bbx = np.array(sorted_bbx)
        classes_detected = sorted_bbx[:, -1].astype(int)

        new_bboxes = np.array([])
        for c in self.classes:
            sub_bboxes = sorted_bbx[classes_detected == c]
            match len(sub_bboxes):
                case 0:
                    # raise CustomError('Error: Can\'t detect ' + self.classes_names + ' . Please, make sure that the uploaded image is a clear and not rotated front of the ID Card image')
                    raise CustomError('Detection_Failure')
                case 1:
                    new_bboxes = np.append(new_bboxes, sub_bboxes[0])
                case _:
                    # most_conf_idx = np.argmax(sub_bboxes[:, -2])
                    # new_bboxes = np.append(new_bboxes, sub_bboxes[most_conf_idx])
                    if c == 2:
                        sub_bboxes = sub_bboxes[sub_bboxes[:, 1] > new_bboxes[1]]
                        sub_bboxes = np.sort(sub_bboxes, axis=-2)
                        box_idx = np.argmin(sub_bboxes[:, 1]-new_bboxes[1])
                    else:
                        box_idx = np.argmax(sub_bboxes[:, -2])
                    new_bboxes = np.append(new_bboxes, sub_bboxes[box_idx])
        
        return new_bboxes.reshape((-1, 6))

    def localize(self, img):
        """
        Localizes objects which are {First Name, Last Name, ID} in an image using a pre-trained model.

        Args:
            img: A NumPy array or the path to an image file to be processed.

        Returns:
            A tensor of bounding boxes for the detected objects.

        """

        # if the path of the image is passed
        if isinstance(img, str):
            img = cv2.imread(img)
        
        preprocessed_img = self.preprocess_image(img)
        cv2.imwrite(self.img_path, preprocessed_img)
        results = self.model.predict(source=self.img_path, save=False, show=False, verbose=False)
        bboxes = results[0].boxes.data

        if len(bboxes) > 3:
            try:
                bboxes = self.filter_bboxes(bboxes)
            except CustomError as e:
                raise CustomError(e.message)
                # print(e.message)
                # return -1
        
        elif len(bboxes) < 3:
            # raise CustomError('Error: Can\'t detect name and id . Please, make sure that the uploaded image is a clear and not rotated front of the ID Card image')
            raise CustomError('Detection_Failure')

        sorted_bbx = sorted(bboxes.tolist(), key=lambda x: x[-1])
        sorted_bbx = np.array(sorted_bbx)
        return sorted_bbx
    
################################################################ End of Text Localizaer ################################################################

################################################################ Start of Text Recognizer ################################################################
class TextRecognizer():
    model = YOLO("code_files/id/ocr/best_recognize.pt")
    reader = Reader(lang_list=['ar'], gpu=True)
    IMG_SIZE = 640
    id_img_path = TMP_PATH + 'tmp_id_img.jpg'
    img_path = TMP_PATH + 'tmp_img.jpg'

    def __init__(self) -> None:
        if not os.path.exists(TMP_PATH):
            os.makedirs(TMP_PATH)
    
    def prepare_bbox(self, bboxes):
        """
        Prepares bounding boxes by sorting them by their x coordinates to read
        the characters from left to right and making the class 
        number of an integer type for further processing.

        Args:
            bboxes: A NumPy array representing the bounding boxes to be prepared.

        Returns:
            A NumPy array representing the prepared bounding boxes.

        """

        # sorted_bbx = sorted(bboxes.tolist(), key=lambda x: x[-1])
        sorted_bbx = bboxes[:, :-2].astype(int)
        return sorted_bbx

    def get_horizontal_list(self, bboxes):
        """
        Converts a list of bounding boxes to a list of horizontal bounding boxes to be passed to easyocr.

        Args:
            bboxes: A NumPy array representing the bounding boxes to be converted.

        Returns:
            A list of horizontal bounding boxes in the format [x_min, x_max, y_min, y_max].
        """

        horizontal_list = []
        for b in bboxes:
            x_min, y_min, x_max, y_max = b
            horizontal_list.append([x_min, x_max, y_min, y_max])
        return horizontal_list

    def recognize_text(self, img_path, horizontal_list):
        """
        Recognizes arabic characters text in an image using easyOCR.

        Args:
            img_path: The path to the image file to be processed.
            horizontal_list: A list of horizontal bounding boxes for the text to be recognized.

        Returns:
            A list of OCR results, each containing the recognized text, its confidence score, and its bounding box.

        """

        img = cv2.imread(img_path)
        ocr_results = self.reader.recognize(img, horizontal_list, [])
        return ocr_results
    
    def format_name(self, ocr_results):
        """
        Formats OCR results for a name by concatinating both the first name and the last name.

        Args:
            ocr_results: A list of OCR results, each containing the recognized text, its confidence score, and its bounding box.

        Returns:
            A formatted string representing the name, in the format "First Name Last Name".

        """

        name = ''
        name = ocr_results[0][-2] + ' ' + ocr_results[1][-2]
        return name

    def recognize_name(self, bboxes):
        """
        Recognizes a name in an image using OCR.

        Args:
            bboxes: A NumPy array representing the bounding boxes for the name to be recognized.

        Returns:
            A string representing the recognized name, in the format "First Name Last Name".

        """

        horizontal_list = self.get_horizontal_list(bboxes)
        ocr_results = self.recognize_text(self.img_path, horizontal_list)
        if len(ocr_results) == 0:
            # raise CustomError("Error: No name was recognized. Please make sure the name section in the uploaded image is clear and the image is not cropped.")
            raise CustomError('Name_Recognition_Failure')
        
        name = self.format_name(ocr_results)
        return name

    def preprocess_id_img(self, bbox):
        """
        Preprocesses an identity card (ID) image by cropping, resizing, and saving the id number part to a file.

        Args:
            bbox: A NumPy array representing the bounding box for the ID image to be processed.

        Returns:
            None

        """
        img = cv2.imread(self.img_path)
        x0, y0, x1, y1 = bbox
        xmin, xmax = [x0, x1] if x0 < x1 else [x1, x0] 
        ymin, ymax = [y0, y1] if y0 < y1 else [y1, y0]
        id_img = img[ymin:ymax, xmin:xmax]
        img2 = cv2.resize(id_img, (self.IMG_SIZE, self.IMG_SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(self.id_img_path, img2)

    def remove_redundant_detections(self, results):
        """
        Removes redundant detections from a list of object detection results.

        Args:
            results: A list of object detection results, each containing a detected object's class label, confidence score, and bounding box.

        Returns:
            A list of object detection results with redundant detections removed.

        """

        predictions =  sorted(results[0].boxes.data.tolist(), key=lambda x: x[0])
        i = 0
        for p in predictions[1:]:
            if i >= len(predictions):
                break

            x_prev = predictions[i][0]
            x_cur = p[0]

            y_prev = predictions[i][1]
            y_cur = p[1]

            # small
            if abs(p[0] - p[2]) < 20 and abs(p[1] - p[3]) < 80:
                if p in predictions: 
                        predictions.remove(p)
                        i += 1

            # overlab case
            if abs(x_cur - x_prev) <= abs(x_cur-p[2])/2:
                if abs(y_prev - y_cur) <= abs(y_cur-p[3])/2:
                    if p[-2] < predictions[i][-2]:
                        if p in predictions: 
                            predictions.remove(p)
                            i += 1
                    else:
                        predictions.remove(predictions[i])
                        i -= 1
                else:
                    y_prev_prev = predictions[i-1][1] if i > 0 else y_prev
                    y_next =  predictions[i+1][1] if i+1 < 13 else y_cur
                    avg_y = (y_next+y_prev_prev) / 2
                    if abs(y_cur - avg_y) > abs(y_cur-p[3]):
                        if p in predictions: 
                            predictions.remove(p)
                            i += 1
                    elif abs(y_prev - avg_y) > abs(y_prev-predictions[i][3]):
                        predictions.remove(predictions[i])
                        i -= 1
            
            # outlier case in which there is a false prediction for a number that's dirt 
            else:
                y_prev_prev = predictions[i-1][1] if i > 0 else y_prev
                y_next =  predictions[i+1][1] if i+1 < 13 else y_cur
                avg_y = (y_next+y_prev_prev) / 2
                if abs(y_cur - avg_y) > abs(y_cur-p[3]):
                    if p in predictions: 
                        predictions.remove(p)
                        i += 1
                elif abs(y_prev - avg_y) > abs(y_prev-predictions[i][3]):
                    predictions.remove(predictions[i])
                    i -= 1
       
            if len(predictions) == 14:
                break

            i += 1 
        return predictions
    
    def recognize_id(self, bbox):
        """
        Recognizes an identity card (ID) number in an image using object detection and OCR.

        Args:
            bbox: A NumPy array representing the bounding box for the ID image to be recognized.

        Returns:
            A string representing the recognized ID number.

        """

        self.preprocess_id_img(bbox)
        results = self.model.predict(source=self.id_img_path, save=False, show=False, verbose=False, conf=0.5)
        pred = results[0].boxes.data.tolist()
        if len(pred) > 14:
            pred = self.remove_redundant_detections(results)
        elif len(pred) < 14:
            # raise CustomError('Error: no id predicted. Please make sure the uploaded image is a clear, non-blurry, and not rotated ID card and the ID part is not occluded.')
            raise CustomError('ID_Recognition_Failure')

        pred_id = ''.join(np.array(sorted(pred, key= lambda x: x[0]))[:, -1].astype(int).astype(str))
        return pred_id

    def recognize(self, localizer_results):
        """
        Recognizes a name and an identity card (ID) number in an image using object detection and OCR.

        Args:
            localizer_results: A list of object detection results, each containing a detected object's class label, confidence score, and bounding box.

        Returns:
            A tuple of two strings, representing the recognized name and ID number, respectively.

        """

        bboxes = self.prepare_bbox(localizer_results)
        try:
            name = self.recognize_name(bboxes[:-1])
        except CustomError as e:
            raise CustomError(e.message)

        try:
            id = self.recognize_id(bboxes[-1])
        except CustomError as e:
            raise CustomError(e.message)

        return name, id
    
class IdOCR():
    l = TextLocalizer()
    r = TextRecognizer()

    def apply_ocr(self, img_path):
        """
        Applies optical character recognition (OCR) to recognize a name and an identity card (ID) number in an image.

        Args:
            img_path: A string representing the file path to the input image.

        Returns:
            A tuple of two strings, representing the recognized name and ID number, respectively.
            If the input image does not contain three detected objects, the function returns an error message.

        """
        try:
            l_results = self.l.localize(img_path)
        except CustomError as e:
            raise CustomError(e.message)

        try:
            name, pred_id = self.r.recognize(l_results)
        except CustomError as e:
            raise CustomError(e.message)
    
        return name, pred_id

# D:\Our_Data\\5bf765c4-6073-4b64-50e9-08d8ba39f77f\Images\Personal_Id.png