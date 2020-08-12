
import cv2
import numpy as np
from typing import List, Tuple


# This function takes an image and finds a face in it, returning the grayscale face detected
def detect_faces(
            image_path: str,
            scale_factor: float = 1.02, 
            min_neighb: int = 5
        ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    
    # Read the image
    image = cv2.imread(image_path)   
    
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




# image_path = 'img/training/Colin/20160618-_IMG7252.jpg'