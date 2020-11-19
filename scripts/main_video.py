# Importing the libraries.
import numpy as np
import cv2
import time


# Reading the input from the file.
video = cv2.VideoCapture('../test_data/traffic-sign-to-test.mp4')

# Preparing Writer variable.
writer = None

# Preparing variables dimensions of the frame.
h, w = None, None

# Loading the file that contains the classes names information.
with open('../data_files/classes.names') as f:
    labels = [line.strip() for line in f]

# Loading the trained configuration file and trained weights.
network = cv2.dnn.readNetFromDarknet('../data_files/ts.cfg',
                                     '../data_files/ts.weights')

# Getting list with names of all layers from YOLO v3 network.
layers_names_all = network.getLayerNames()


# Getting only output layers' names that are unconnected and will help us in prediction.
layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

# Setting Minimum Probability factor to eliminate weak predictions.
probability_minimum = 0.5

# Setting threshold for Non-Maximum Supression to filter weak bounding boxes.
threshold = 0.3

# Generating colours for representing every detected object.
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# Create frame and time counter variables.
f=0; t=0

# Create an OpenCV resizable window.
cv2.namedWindow("Result_Video", cv2.WINDOW_NORMAL)

# Start of reading frames.
while True:
    
    # Read the frame and return value.
    ret, frame = video.read()
    
    # If return value is False (i.e end of video, break).
    if not ret:
        break
    
    # If w or h is not initialized initialize them.
    if w is None or h is None:
        h, w = frame.shape[:2]

    # Create the blob from the found frame and resize and scale it.
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)


    # Implementing the forward prop only using our output layers and setting blob as input.
    network.setInput(blob)  
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    # Increasing counters for frames and total time.
    f += 1
    t += end - start

    # Showing spent time for single current frame.
    print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))

    # Preparing lists for detected bounding boxes,
    
    bounding_boxes = []
    confidences = []
    class_numbers = []

    # Going through all output layers after feed forward pass
    for result in output_from_network:
        # Going through all detections from current output layer
        for detected_objects in result:
            # Getting 80 classes' probabilities for current detected object
            scores = detected_objects[5:]
            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)
            # Getting value of probability for defined class
            confidence_current = scores[class_current]

            # # Check point
            # # Every 'detected_objects' numpy array has first 4 numbers with
            # # bounding box coordinates and rest 80 with probabilities
            #  # for every class
            # print(detected_objects.shape)  # (85,)

            # Eliminating weak predictions with minimum probability
            if confidence_current > probability_minimum:
                
                #Converting data into YOLO v3 format.
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Adding results into prepared lists
                bounding_boxes.append([x_min, y_min,
                                       int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)

    # If any one object is detected after Non-maximum supresison.
    if len(results) > 0:
        # Going through indexes of results
        for i in results.flatten():
            # Getting current bounding box coordinates, width and height.
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            # Preparing colour for current bounding box
            colour_box_current = colours[class_numbers[i]].tolist()

            # Drawing bounding box on the original current frame
            cv2.rectangle(frame, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)

            # Preparing text with label and confidence for current bounding box
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                   confidences[i])

            # Putting text with label and confidence on the original image
            cv2.putText(frame, text_box_current, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

    if writer is None:
        
        # Creating the codecc for the wirter.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Writing current frame into the writer.
        writer = cv2.VideoWriter('../result_data/result-traffic-cars.mp4', fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

    # Write processed current frame to the file
    writer.write(frame)

cv2.destroyWindow("Result_Video")


# Printing final results.
print()
print('Total number of frames', f)
print('Total amount of time {:.5f} seconds'.format(t))
print('FPS:', round((f / t), 1))


# Releasing video reader and writer.
video.release()
writer.release()