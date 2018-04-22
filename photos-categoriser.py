
# Libraries
import os
import errno
import cv2
import itertools


# This function will delete a file if it exists, taken from https://stackoverflow.com/questions/10840533/most-pythonic-way-to-delete-a-file-which-may-not-exist
def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred


# This function takes an image and finds a face in it, returning the grayscale face detected
def detect_face(image, scale_factor, min_neighb, verbose = False):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     
    # Use Local Binary Patterns (LBP) histogram, this doesn't care about light/shadow in images but instead only looks at gradient changes in light levels
    face_cascade = cv2.CascadeClassifier('C:/ProgramData/Anaconda3/Library/etc/lbpcascades/lbpcascade_frontalface_improved.xml')
     
    # Detect faces regardless of scale, returns list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor = scale_factor, minNeighbors = min_neighb);
     
    # Return None if no face detected
    if (len(faces) == 0):
        if verbose:
            print('No face detected in this image.')
        return None, None
     
    # Get the face details from the first face found
    (x, y, w, h) = faces[0]
     
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]


# This function will read all persons' training images, detect face from each image and will return two lists of exactly same size, one list of faces and another list of labels for each face
# Modified from https://www.superdatascience.com/opencv-face-recognition/
def prepare_training_data(data_folder_path, scale_factor, min_neighb, save_faces = False, verbose = False):
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
            if '_face.' in image_name:
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
            face, rect = detect_face(image, scale_factor, min_neighb, verbose)
            
            if rect is not None:
                if verbose or save_faces:
                    (x, y, w, h) = rect
                    image_with_rect = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            
                if verbose:                
                    cv2.imshow('Image showing face...', image_with_rect)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                
                if save_faces:
                    output_file_name = os.path.join(subject_dir_path, image_name.replace('.', '_face.'))
                    silentremove(output_file_name)
                    cv2.imwrite(output_file_name, image_with_rect)
            
            # If a face is detected then add it to output
            if face is not None:
                # Generate a numeric label
                label = next(incrementer)
                
                # Add to output
                output.append({'face': face,
                               'label': label,
                               'rect': rect,
                               'image_name': image_name})
    
    return output


# Main program
faces_array = prepare_training_data(data_folder_path = 'C:/GitWorkspace/photos-categoriser/img/training',
                                    scale_factor = 1.02,
                                    min_neighb = 5,
                                    save_faces = True,
                                    verbose = False)



"""
cv2.imshow("Training on image...", faces_array[18]['face'])
cv2.waitKey(0)
cv2.destroyAllWindows()
"""



















