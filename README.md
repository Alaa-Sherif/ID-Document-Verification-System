# ID-Documents-Verification-System
ITI Graduation Project - ID and Document Verification System

## Problem Statement
The problem identified in the ITI’s system, while applying to any of the provided training programs, is <b>the significant amount of time</b> consumed during the document review process.<br> 
As shown in this example below:👇

![image](https://github.com/Alaa-Sherif/ID-Document-Verification-System/assets/43891138/757c7689-280e-4e9d-938a-aabdeb52f08b)


Currently, applicants wait for a long time until their uploaded documents get reviewed because the revision process is done manually by a person. This manual review process is time-consuming. Consequently, it causes delays in the overall application process, particularly when there are a large number of applicants.

## Proposed Solution
We decided to partially automate this part of the system to save a lot of time. Hence, we used machine learning techniques, such as classification and OCR, to check document validity, extract data from the national ID, etc.

## Project Pipeline
![image](https://github.com/Alaa-Sherif/ID-Document-Verification-System/assets/43891138/fc17cfcb-6575-4d7a-8b92-f9fdf9a9342d)

We apply some preprocessing to all the documents and then check every document separately.

  * <b>Personal image check</b>: we check whether it is a formal image that can be accepted or not based on some defined rules, if it is not valid then we reject it.
  * <b>Document Classification</b>: we check whether the applicant uploaded the required document and if yes, then we extract the required data from it like the ID number.




### To download the weights and the models used in this project, scan this QR Code 👇
<img src="https://github.com/Alaa-Sherif/ID-Document-Verification-System/assets/43891138/4a8a2c26-baad-4406-abc2-30d0d6611e92" width=250 height=250>

### To watch a demo of the project, scan this QR Code 👇
<img src="https://github.com/Alaa-Sherif/ID-Document-Verification-System/assets/43891138/6812937c-68fe-4c95-a435-de56aafc3b08" width=250 height=250>
