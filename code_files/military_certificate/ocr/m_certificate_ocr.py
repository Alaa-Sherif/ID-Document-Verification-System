import cv2
import pytesseract

def catchNormal(text)  :
    if "نهائياً" in text or "النهائى"in text: 
        return "permenant"
    if "مؤقتا" in text or "المؤقت" in text : 
        return "temp"
    return "not decided"

def catch(text)  :
    normal  = catchNormal(text)
    if  normal != "not decided" :
        return normal
    text = text.split(" ")
    for word in text : 
        p = jaccard_similarity(word , "النهائي")
        t = jaccard_similarity(word , "مؤقت")
        if p > 0.5 :
            return "permenant"
        elif t > 0.5 :
            return "temp"

    return "not decided"

def jaccard_similarity(word1, word2):
    set1 = set(word1)
    set2 = set(word2)
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    similarity = intersection / union
    return similarity

def prepro(image , contrast = False , show_text=False) : 
    # image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if contrast == True :
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

    tesseract_config = '--psm 6 --oem 3 -l ara'
    text = pytesseract.image_to_string(image, config=tesseract_config)
    if show_text == True :
        print(text)

    return catch(text)
