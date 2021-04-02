# Importing the libraries.
import numpy as np
import cv2
import time
import tensorflow.keras as keras
import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Path to save and load images.
path_to_test_images = '../test_data/images/'
path_to_save = '../result_data/images/'
    
# Load the trained keras model.
model = keras.models.load_model('../data_files/traffic_classifier.h5')

# Labels of the model.
model_labels = ['Speed limit (20km/h)',
 'Speed limit (30km/h)',
 'Speed limit (50km/h)',
 'Speed limit (60km/h)',
 'Speed limit (70km/h)',
 'Speed limit (80km/h)',
 'End of speed limit (80km/h)',
 'Speed limit (100km/h)',
 'Speed limit (120km/h)', 'No passing',
 'No passing for vehicles over 3.5 metric tons',
 'Right-of-way at the next intersection',
 'Priority road',
 'Yield',
 'Stop',
 'No vehicles',
 'Vehicles over 3.5 metric tons prohibited',
 'No entry',
 'General caution',
 'Dangerous curve to the left',
 'Dangerous curve to the right',
 'Double curve',
 'Bumpy road',
 'Slippery road',
 'Road narrows on the right',
 'Road work',
 'Traffic signals',
 'Pedestrians',
 'Children crossing',
 'Bicycles crossing', 'Beware of ice/snow',
 'Wild animals crossing',
 'End of all speed and passing limits',
 'Turn right ahead',
 'Turn left ahead',
 'Ahead only',
 'Go straight or right',
 'Go straight or left',
 'Keep right',
 'Keep left',
 'Roundabout mandatory',
 'End of no passing',
 'End of no passing by vehicles over 3.5 metric tons']

# Load the names of the classes.
with open('../data_files/classes.names') as f:
    labels = [line.strip() for line in f]

# Load the darnket network
network = cv2.dnn.readNetFromDarknet('../data_files/ts.cfg',
                                     '../data_files/ts.weights')

# Getting list with names of all layers from YOLO v3 network
layers_names_all = network.getLayerNames()

layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

# Setting the porbability threshold.
probability_minimum = 0.5

# Setting the non-maximum supression threshold.
threshold = 0.3

# Generate countours.
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# Find all test images.
for img in os.listdir(path_to_test_images):
    
    # Read the image in BGR format.
    image_BGR = cv2.imread(path_to_test_images+img)
    
    # Dimensions of the BGR image.
    h, w = image_BGR.shape[:2]
    
    
    
   # Creating the blob from the current image.
    blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)

    # Start forward pass using blob as the input
    network.setInput(blob)
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()
    
    # Showing spent time for forward pass
    print('Objects Detection took {:.7f} seconds'.format(end - start))
    
    # Preparing list for bounding boxes.
    bounding_boxes = []
    confidences = []
    class_numbers = []
    
    
    # Going through all output layers after feed forward pass
    for result in output_from_network:
        # Going through all detections from current output layer
        for detected_objects in result:
            # Detected_object has first 4 elements as x_center, y_center, h, w of the current bounding box
            scores = detected_objects[5:]
            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)
            # Getting value of probability for defined class
            confidence_current = scores[class_current]
        
            # Eliminating weak predictions with minimum probability
            if confidence_current > probability_minimum:
                
                # COnverting the found bounding boxes into yolo-format.
                
                box_current = detected_objects[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))
    
                # Adding results into prepared lists
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)
    
    # Implementing non-maximum supression.
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)
    
  
    # Defining counter for detected objects
    counter = 1
    
    # Checking if there is at least one detected object after non-maximum suppression
    if len(results) > 0:
        # Going through indexes of results
        for i in results.flatten():
            # Showing labels of the detected objects
            print('Object {0}: {1}'.format(counter, labels[int(class_numbers[i])]))
    
            # Incrementing counter
            counter += 1
    
            # Getting current bounding box coordinates,
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
    
           # Finding colour of the current bounding box.
            colour_box_current = colours[class_numbers[i]].tolist()
    
            # Drawing bounding box on the original image
            cv2.rectangle(image_BGR, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)
            
             # Find out the label of the current frame.
            roi = image_BGR[y_min : y_min+box_height, x_min : x_min+box_width] 
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = cv2.resize(roi,(30, 30),interpolation=cv2.INTER_CUBIC)
            roi = keras.preprocessing.image.img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            pred = model.predict(roi)
            found_label = model_labels[ np.argmax(pred) ]
    
            # Putting text with label and confidence on the original image.
            cv2.putText(image_BGR, found_label, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)
    
    
    # Printing the number of predections detected.
    cv2.imwrite(path_to_save+img, image_BGR)
    print()
    print('Total objects been detected:', len(bounding_boxes))
    print('Number of objects left after non-maximum suppression:', counter - 1)
       
