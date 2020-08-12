
# Basic imports
from typing import Dict
import numpy as np

# Tensorflow
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


def detect_objects(
            image_path: str, 
            min_prob_filter: float = 0.1
        ) -> Dict[str, float]:
    
    # Set the shape the image needs to be for VGG19 model
    input_shape = (224, 224)
    
    # Convert the image into the way we want it
    image = load_img(image_path, target_size = input_shape)
    image = img_to_array(image)
    
    # Image is loaded as shape (inputShape[0], inputShape[1], 3) however we need to expand the
    # dimension by making the shape (1, inputShape[0], inputShape[1], 3) so we can pass it through
    # the network, this is equivalent to adding a batch (which is how the model was trained)
    image = np.expand_dims(image, axis = 0)
    
    # Preprocess the image by mean subtraction
    image = imagenet_utils.preprocess_input(image)
    
    # Get the neural network and download the relevant weights (include top specifies to include the final classification layer)
    nn_model = VGG19(weights = 'imagenet', include_top = True)
    
    # Make a prediction and decode
    predictions_matrix = nn_model.predict(image)
    predictions = imagenet_utils.decode_predictions(predictions_matrix)
    
    # Iterate over results and keep relevant ones (predictions[0] as we want the first batch result, since we made one batch for predicting)
    object_probabilities = {}
    for code, object_name, prob in predictions[0]:
        if prob >= min_prob_filter:
            object_probabilities[object_name] = prob
    
    return object_probabilities


# image_path = 'img/training/Colin/20160618-_IMG7252.jpg'