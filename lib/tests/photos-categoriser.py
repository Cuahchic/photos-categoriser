
# Libraries
from typing import List, Tuple, Any, Dict
import os
import errno
import cv2
import itertools
import numpy as np


# This function will delete a file if it exists, taken from https://stackoverflow.com/questions/10840533/most-pythonic-way-to-delete-a-file-which-may-not-exist
def silentremove(filename: str) -> None:
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred


# This function takes an image and finds a face in it, returning the grayscale face detected
def detect_face(
            image: np.ndarray,
            scale_factor: float, 
            min_neighb: int, 
            verbose: bool = False
        ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     
    # Use Local Binary Patterns (LBP) histogram, this doesn't care about light/shadow in images but instead only looks at gradient changes in light levels
    face_cascade = cv2.CascadeClassifier('config/lbpcascade_frontalface_improved.xml')
    
    # Get min size
    min_size = (int(image.shape[0] / 10), int(image.shape[1] / 10))
     
    # Detect faces regardless of scale, returns list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor = scale_factor, minNeighbors = min_neighb, minSize = min_size)
     
    # Return None if no face detected
    if (len(faces) == 0):
        if verbose:
            print('No face detected in this image.')
        return None, None
    
    # Get the outputs
    faces_out = []      # List of image snippets showing the face detected
    rects_out = []      # List of rectangles around each face
    for face in faces:
        # Get the face details from the first face found
        (x, y, w, h) = face
        
        # Add to lists
        faces_out.append(face)
        rects_out.append(gray[y:y+w, x:x+h])
     
    #return only the face part of the image
    return rects_out, faces_out


# This function will read all persons' training images, detect face from each image and will return two lists of exactly same size, one list of faces and another list of labels for each face
# Modified from https://www.superdatascience.com/opencv-face-recognition/
def prepare_training_data(
            data_folder_path: str,      # Top level folder which contains subfolders labelled per person
            scale_factor: float,        # How much the image size is reduced at each image scale 
            min_neighb: int,            # How many neighbors each candidate rectangle should have to retain it
            save_faces: bool = False,   # Whether to save the detected face(s) in an accompanying image in the same directory
            verbose: bool = False       # Whether to print progress as it goes along
        ) -> List[Dict[str, Any]]:
    
    # List all subfolders, which should be of different people
    dirs = os.listdir(data_folder_path)
     
    # Outputs
    output = []
 
    # let's go through each directory and read images within it
    incrementer = itertools.count()
    for dir_name in dirs:
        # Get person level folder of training images
        subject_dir_path = os.path.join(data_folder_path, dir_name)
        subject_images_names = os.listdir(subject_dir_path)
         
        # For each image check if it contains a face and if so add it to the output
        for image_name in subject_images_names:
            # If the image is one of the saved outputs showing the face square then skip it
            if '_allfaces.' in image_name or 'face.' in image_name:
                continue
            
            # Get path to image and read it in
            image_path = os.path.join(subject_dir_path, image_name)
            image = cv2.imread(image_path)

            # Optionally briefly flash up the image         
            if verbose:
                cv2.imshow('Training on image...', image)
                cv2.waitKey(100)
                cv2.destroyAllWindows()
 
            # Detect whether there is a face in the image
            faces, rects = detect_face(image, scale_factor, min_neighb, verbose)
            
            if rects is not None:
                if verbose or save_faces:
                    image_with_rect = image
                    for rect in rects:
                        (x, y, w, h) = rect
                        image_with_rect = cv2.rectangle(image_with_rect, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            
                if verbose:                
                    cv2.imshow('Image showing face...', image_with_rect)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                
                if save_faces:
                    # Save main image with all faces
                    output_file_name = os.path.join(subject_dir_path, image_name.replace('.', '__allfaces.'))
                    silentremove(output_file_name)
                    cv2.imwrite(output_file_name, image_with_rect)
                    
                    # Save individual faces
                    for i, face in enumerate(faces):
                        output_file_name = os.path.join(subject_dir_path, image_name.replace('.', f'_{i}face.'))
                        silentremove(output_file_name)
                        cv2.imwrite(output_file_name, face)
            
            # If a face is detected then add it to output
            if faces is not None:
                # Generate a numeric label
                label = next(incrementer)
                
                # Add to output
                output.append({'face': faces,
                               'label': label,
                               'rect': rects,
                               'image_name': image_name})
    
    return output


# Main program
faces_array = prepare_training_data(
                    data_folder_path = 'img/training',
                    scale_factor = 1.02,
                    min_neighb = 5,
                    save_faces = True,
                    verbose = False
                )





















