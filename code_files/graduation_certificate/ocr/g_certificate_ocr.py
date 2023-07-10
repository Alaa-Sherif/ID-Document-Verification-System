import requests
import re
import flair
from flair.models import SequenceTagger

def recognize_text(image_path, api_key):
    payload = {
        'apikey': api_key,
        'language': 'ara',  # Set the language to Arabic
    }
    with open(image_path, 'rb') as image_file:
        files = {'file': image_file}
        response = requests.post('https://api.ocr.space/parse/image', data=payload, files=files)

    if response.status_code == 200:
        result = response.json()
        return result
    else:
        return None
    
def OCR(image_path):
    result = recognize_text(image_path, 'K83256134988957')
    if result and result['IsErroredOnProcessing'] is False:
        parsed_text = result['ParsedResults'][0]['ParsedText']            
        return parsed_text

#extract persons arabic names 
def extract_arabic_names(parsed_text):
    tagger = SequenceTagger.load("megantosh/flair-arabic-multi-ner")

    # Perform NER on the parsed text
    sentences = flair.data.Sentence(parsed_text)
    tagger.predict(sentences)
    
    # Extract Arabic names
    arabic_names = []
    for entity in sentences.get_spans('ner'):
        if entity.tag == 'PER':
            arabic_names.append(entity.text)
    
    return arabic_names

def extract_information(ocr_result):
    ## Remove non-alphanumeric characters
    print(ocr_result)
    ocr_result = re.sub(r'[^a-zA-Z0-9\u0600-\u06FF\s:/]+', '', ocr_result)

    # Define patterns for university name, person's name, and grade
    university_pattern = r'(جامعة [^ ]+|كلية [^ ]+|المعهد[^ ]+)'
    person_name_pattern = r'(?:السيد|الطالبة|الطالب):?\s*(?:/)?\s*([\u0600-\u06FF\s]+)'
    grade_pattern = r'(?:بتقدير عام|التقدير العام :|:بتقدير|تقدير عام:|تقدير:) ((?:جيد جدا|ممتاز مع مرتبة الشرف|امتياز مع مرتبة الشرف|جيد جدا|امتياز|امتياز مع مرتبة الشرف|جيد|جيد مع مرتبة الشرف))'

    # Initialize variables
    university_name = ''
    person_name = ''
    grade = ''

    # Extract information
    university_match = re.search(university_pattern, ocr_result)
    if university_match:
        university_name = university_match.group(0)

    person_match = re.search(person_name_pattern, ocr_result)
    if person_match:
        person_name = person_match.group(1)
    else:
      arabic_names = extract_arabic_names(ocr_result)
      person_name = arabic_names[0] if arabic_names else None

    grade_match = re.search(grade_pattern, ocr_result)
    if grade_match:
        grade = grade_match.group(1)

    return university_name, person_name, grade
