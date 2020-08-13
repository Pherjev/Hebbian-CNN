import cv2
import numpy as np
#from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
#from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.models import Model
#import matplotlib.pyplot as plt

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('horse11.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

c = 0

#model = VGG19()
#model.layers.pop()
#model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

F = []

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  c += 1
  print(c)
  ret, frame = cap.read()
  if ret == True:

    # Display the resulting frame
    frame = cv2.resize(frame,(224,224))
    cv2.imshow('Frame',frame)
    if c < -1:
      frame = img_to_array(frame)
      frame = frame.reshape((1, frame.shape[0], frame.shape[1], frame.shape[2]))
      frame = preprocess_input(frame)
      features = model.predict(frame, verbose=0)
      print(features)
      F.append(features)
      #print(np.argmax(features))
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

    if c == 200:
      print("Lizzy")
      break


    #plt.plot(len(features),features)
    #plt.pause(0.05)

  # Break the loop
  else: 
    break


#F = np.array(F)

#np.save('airplane1.npy',F)

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
