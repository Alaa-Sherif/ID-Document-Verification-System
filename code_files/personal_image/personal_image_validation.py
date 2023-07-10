import cv2
import mediapipe as mp
import math
import numpy as np
import os


def save_image_to_folder(image, folder_path, image_name):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Create the full path for the image
    image_path = f"{folder_path}/{image_name}"
    
    # Save the image to the specified path
    cv2.imwrite(image_path, image)
    print(f"Image saved successfully to {image_path}")


def face_size(result , shape):

    face_detection_result = result.detections[0]
    bbox = face_detection_result.location_data.relative_bounding_box

    # Get the width and height of the face bounding box
    width = bbox.width * shape[1]
    height = bbox.height * shape[0]

    return width , height



def face_ratio(width , height , shape):

    # Calculate the ratios
    width_ratio = width / shape[1]
    height_ratio = height / shape[0]

    return width_ratio , height_ratio



def detect_faces(image):
    
    # Initialize the face detection module
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection()
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    
    # Detect faces in the image
    results = face_detection.process(image)

    # Get the face landmarks
    landmarks_results = face_mesh.process(image)

    return results , landmarks_results



def rectify_rotated_face(image , results , landmarks_results):

    # Get the first face detection
    face_detection_result = results.detections[0]
    bbox = face_detection_result.location_data.relative_bounding_box


    # Get the first face landmarks
    face_landmarks = landmarks_results.multi_face_landmarks[0]
    
    # Calculate the rotation angle of the face
    left_eye = (face_landmarks.landmark[33].x,
                face_landmarks.landmark[33].y)
    right_eye = (face_landmarks.landmark[263].x,
                 face_landmarks.landmark[263].y)

    # Calculate the angle between the eyes
    angle_radians = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
    angle_degrees = math.degrees(angle_radians)

   # Rotate the image
    center_x = int((bbox.xmin + bbox.width / 2) * image.shape[1])
    center_y = int((bbox.ymin + bbox.height / 2) * image.shape[0])
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle_degrees, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    # Convert the rotated image back to RGB
    rotated_image_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    rotated_results , _ = detect_faces(rotated_image_rgb)

    # Get the first face detection
    face_detection_result = rotated_results.detections[0]
    bbox = face_detection_result.location_data.relative_bounding_box

    # Calculate the bounding box coordinates in pixels
    x = int(bbox.xmin * rotated_image.shape[1])
    y = int(bbox.ymin * rotated_image.shape[0])
    width = int(bbox.width * rotated_image.shape[1])
    height = int(bbox.height * rotated_image.shape[0])

    # Crop the image based on the bounding box
    cropped_image = rotated_image[y:y+height, x:x+width]

    cropped_image_1d = cropped_image.flatten()

    cropped_image_mean = np.mean(cropped_image_1d)

    # Return the rectified face image
    return rotated_image , cropped_image_mean





def delete_extra_zeros(image):
    
    b , g , r = cv2.split(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for th in [0.8,0.1]:

        # Calculate the threshold for number of zeros in each column
        threshold_count = gray_image.shape[0] * th

        # Find columns that have more than threshold_count zeros
        cols_with_zeros = np.sum(gray_image == 0, axis=0) > threshold_count

        # Delete columns with excessive zero values
        gray_image = gray_image[:, ~cols_with_zeros]
        b = b[:, ~cols_with_zeros]
        g = g[:, ~cols_with_zeros]
        r = r[:, ~cols_with_zeros]



        threshold_count = gray_image.shape[1] * th

        # Find rows that have more than threshold_count zeros
        rows_with_zeros = np.sum(gray_image == 0, axis=1) > threshold_count

        # Delete rows with excessive zero values
        gray_image = gray_image[~rows_with_zeros, :]
        b = b[~rows_with_zeros, :]
        g = g[~rows_with_zeros, :]
        r = r[~rows_with_zeros, :]



    new = cv2.merge((b , g , r))
    return new



def find_farthest_from_mean(arr):
    # Calculate the mean of the array
    mean = np.mean(arr)

    # Calculate the absolute difference of each element from the mean
    differences = np.abs(arr - mean)

    # Find the index of the element with the maximum difference
    farthest_index = np.argmax(differences)

    return farthest_index



def count_zeros(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Count the number of zero pixels
    num_zeros = np.sum(gray == 0)

    return num_zeros


def face_angle(image , landmarks_results):

    img_h, img_w, _ = image.shape
    face_3d = []
    face_2d = []

    if not landmarks_results.multi_face_landmarks:
        return True

    for idx, lm in enumerate(landmarks_results.multi_face_landmarks[0].landmark):
        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
            if idx == 1:
                nose_2d = (lm.x * img_w, lm.y * img_h)
                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

            x, y = int(lm.x * img_w), int(lm.y * img_h)

            # Get the 2D Coordinates
            face_2d.append([x, y])

            # Get the 3D Coordinates
            face_3d.append([x, y, lm.z])       
            
    # Convert it to the NumPy array
    face_2d = np.array(face_2d, dtype=np.float64)

    # Convert it to the NumPy array
    face_3d = np.array(face_3d, dtype=np.float64)

    # The camera matrix
    focal_length = 1 * img_w

    cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                            [0, focal_length, img_w / 2],
                            [0, 0, 1]])

    # The distortion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    # Solve PnP
    _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

    # Get rotational matrix
    rmat, _ = cv2.Rodrigues(rot_vec)

    # Get angles
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    # Get the y rotation degree
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360

    # print(x,y)
    # See if the user's head tilting
    if y < -10 or y > 10 or x < -10 or x > 10:
        
        return True
    
    return False


def single_face_mesh(image, folder):
    # image = cv2.imread(image_path)
    # image = cv2.rotate(image, cv2.ROTATE_180)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_name = folder + '.png'
    

    candidate_images = []
    faces_maen = []
    first_detected_face = True

    for i in range(4):
        results , landmarks_results = detect_faces(image_rgb)

        # Get the number of faces detected
        num_faces = results.detections
        
        if landmarks_results.multi_face_landmarks:
            
            if len(num_faces) == 1:

                # cv2.imshow('detected face', cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                if first_detected_face:
                    # face size
                    width , height = face_size(results,image.shape)
                    # print(f"Face Width: {width}px")
                    # print(f"Face Height: {height}px")

                    if width < 200 or height < 200 :
                        # save_image_to_folder(image,'output/personal_image/Pixelated_Face', image_name)
                        return "Pixelated_Face" , image

                    # face ratio
                    width_ratio , height_ratio = face_ratio(width , height , image.shape)
                    # print(f"Face Width Ratio: {width_ratio:.2f}")
                    # print(f"Face Height Ratio: {height_ratio:.2f}")

                    if width_ratio < 0.3 or height_ratio < 0.25 :
                        # save_image_to_folder(image,'output/personal_image/Too_Small_Face', image_name)
                        return "Too_Small_Face" , image
                    
                    # face resolution
                    # if blurry_check(image_rgb):
                    #     save_image_to_folder(image,'output/personal_image/Poor_Image_Resolution', image_name)
                    #     return "Poor_Image_Resolution"
                    
                    first_detected_face = False

                # rectify the image
                rectified_image , face_mean = rectify_rotated_face(image_rgb , results , landmarks_results)

                candidate_images.append(rectified_image)
                faces_maen.append(face_mean)

                # cv2.imshow('detected face', cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))
                # cv2.imshow('detected face', cv2.cvtColor(rectified_image, cv2.COLOR_BGR2RGB))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

            else:
                # save_image_to_folder(image,'output/personal_image/Multiple_People', image_name)
                return "Multiple_People" , image
            
        image_rgb = cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)
    
    if len(faces_maen) == 0:
        # save_image_to_folder(image,'output/personal_image/No_Faces_Detected', image_name)
        return "No_Faces_Detected" , image
    
    elif len(faces_maen) == 4:
        idx = find_farthest_from_mean(faces_maen)
        del candidate_images[idx]
    
    idx  = np.argmin(np.array([count_zeros(x) for x in candidate_images]))

    finalized_image = delete_extra_zeros(candidate_images[idx])

    finalized_image_rgb = cv2.cvtColor(finalized_image, cv2.COLOR_BGR2RGB)

    _ , landmarks_results = detect_faces(finalized_image_rgb)

    if landmarks_results.multi_face_landmarks and len(landmarks_results.multi_face_landmarks[0].landmark) != 468:
        return "unalligned face with the camera" # working on it to centralize the face

    if face_angle(finalized_image_rgb , landmarks_results):
        # save_image_to_folder(image,'output/personal_image/Not_Facing_The_Camera', image_name)
        return "Not_Facing_The_Camera" , image

    # cv2.imshow('detected face', finalized_image_rgb)
    # cv2.imshow('original', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(len(candidate_images),faces_maen)

    # save_image_to_folder(finalized_image_rgb,'output/personal_image/Accepted', image_name)
    return "Accepted" , finalized_image_rgb




