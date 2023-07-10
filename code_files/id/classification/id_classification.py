import cv2
import numpy as np

def ID_Classifier(org, model):
        
    # resize the image
    image = cv2.resize(org, (224, 224), interpolation=cv2.INTER_LINEAR)

    # normalize the image
    image = np.array(image).astype(np.float32) / 255

    # expand the image dimention to be able to make the prediction
    image = np.expand_dims(image, axis=0)

    # classify ID images
    pred = model.predict(image)
    pred = np.round(pred)
    label = (np.where(pred==1)[1])

    if label == 0:
        image_label = 'Cropped'
    elif label == 1:
        image_label = 'ID'
    elif label == 2:
        image_label = 'Other'
    elif label == 3:
        image_label = 'Rotated'
    else:
        image_label = 'Vertical FB'
            
    return image_label

#-------------------------------------------------------------------------------------------------------------------
# it takes an image as a numpy array
def classify_ID_img(image, model):

    predicted_labels = []

    # classify the image
    pred_label = ID_Classifier(image, model)

    # if the image with front & back then split it
    if pred_label == 'Vertical FB':
        front = image[:image.shape[0]//2]
        back = image[image.shape[0]//2:image.shape[0]]

        front_label = ID_Classifier(front, model)
        back_label = ID_Classifier(back, model)

        predicted_labels.append(front_label)
        predicted_labels.append(back_label)
    
    else:
        predicted_labels.append(pred_label)

    return predicted_labels
