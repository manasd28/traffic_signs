# Importing the libraries.
import numpy as np
import cv2
import time
import keras

# Reading the input from the primary camera.
camera = cv2.VideoCapture(0)

# Variables for Dimension of frames.
h, w = None, None

# Opening and loading the lables of the Yolo Classes.
with open('../data_files/classes.names') as f:
    labels = [line.strip() for line in f]

# Loading the trained CNN model.
model = keras.models.load_model('../data_files/traffic_classifier.h5')

# Getting the model labels.
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

# Reading the yolo darknet network.
network = cv2.dnn.readNetFromDarknet('../data_files/ts.cfg',
                                     '../data_files/ts.weights')

# Getting the list of all the layers of the network.
layers_names_all = network.getLayerNames()

# Getting the name of all the layers.
layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

# Settting minimum probability threshold to eleminate weak predictions.
probability_minimum = 0.5

# Setting threshold for non - maximum supression.
threshold = 0.3

# Generating colours for the bounding boxes.
colours = np.random.randint(0, 255, size=(len(model_labels), 3), dtype='uint8')


# Start reading the frames.
while True:
    
    _, frame = camera.read()

    # Finding the dimensions of the frame.
    if w is None or h is None:
        # Slicing from tuple only first two elements
        h, w = frame.shape[:2]

    # Creating an blog from the given frame & swapping the R and B channels   
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    
    # Starting the input of the forward pass
    network.setInput(blob)
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    # Showing the time of processing of the each frame
    print('Current frame took {:.5f} seconds'.format(end - start))

    
    # Preparing the lists for bounding_boxes, confidences and class_numbers.
    bounding_boxes = []
    confidences = []
    class_numbers = []

    # Going through all output layers after feed forward pass
    for result in output_from_network:
        
        # Going through all detections from current output layer
        for detected_objects in result:
            
            # Getting the probability scores of the classes
            scores = detected_objects[5:]
            
            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)
            
            # Getting value of probability for defined class
            confidence_current = scores[class_current]


            # Eliminating weak predictions with minimum probability
            if confidence_current > probability_minimum:
                
                # Getting the bounding boxes of the detected objects in yolo format.
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                # Find the width and the height.
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Adding results into prepared lists
                bounding_boxes.append([x_min, y_min,
                                       int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

     
    # Implementing non-maximum supression for the given bounding boxes.
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)

    # Checking objects after non-maximum supression.
    if len(results) > 0:
        
        # Going through indexes of results
        for i in results.flatten():
            
            # Getting the current frame from bounding boxes.
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            # Preparing colour for the current bounding box.
            colour_box_current = colours[class_numbers[i]].tolist()

            # Drawing the rectangle on the found frame.
            cv2.rectangle(frame, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 4)

            # Find out the label of the current frame.
            roi = frame[y_min : y_min+box_height, x_min : x_min+box_width] 
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = cv2.resize(roi,(30,30),interpolation=cv2.INTER_CUBIC)
            roi = keras.preprocessing.image.img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            pred = model.predict(roi)
            found_label =  model_labels[np.argmax(pred)]

            # Putting text with label and confidence on the original image
            cv2.putText(frame, found_label, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

    # Creating a resizable window.
    cv2.namedWindow('YOLO v3 Real Time Detections', cv2.WINDOW_NORMAL)
    cv2.imshow('YOLO v3 Real Time Detections', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Releasing the camera and destroying all frames.
camera.release()
cv2.destroyAllWindows()
