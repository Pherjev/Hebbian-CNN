import cv2
import numpy as np
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import preprocess_input

#import matplotlib.pyplot as plt

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name



model = Xception()
model.layers.pop()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
model.summary()

for num in range(15): # 15
  num += 1
  name = 'ship' + str(num)

  cap = cv2.VideoCapture(name + '.mp4')

  # Check if camera opened successfully
  if (cap.isOpened()== False): 
    print("Error opening video stream or file")

  c = 0


  F = []

  # Read until video is completed
  while(cap.isOpened()):
    # Capture frame-by-frame
    c += 1
    ret, frame = cap.read()
    if ret == True:

      # Display the resulting frame
      frame = cv2.resize(frame,(299,299))
      #cv2.imshow('Frame',frame)
      if c <= 200:
        frame = img_to_array(frame)
        frame = frame.reshape((1, frame.shape[0], frame.shape[1], frame.shape[2]))
        frame = preprocess_input(frame)
        features = model.predict(frame, verbose=0)
        print(features,c)
        F.append(features)
        #print(np.argmax(features))
      else:
        break
      # Press Q on keyboard to  exit
      #if cv2.waitKey(25) & 0xFF == ord('q'):
      #  break

      #plt.plot(len(features),features)
      #plt.pause(0.05)

    # Break the loop
    else: 
      break


  F = np.array(F)

  np.save(name + '.npy',F)

  #for i in range(len(F)):
  #  plt.plot(len(F[i]),F[i])
  #  plt.show()

  print("c=" + str(c))
  # When everything done, release the video capture object
  cap.release()

  # Closes all the frames
  cv2.destroyAllWindows()

# REFERENCIAS

#https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
