"""
Program to implement the trained model in Real-World Environment
""" 
#Import the necessary modules
import h5py
from keras.models import Model,load_model
from keras.models import model_from_json
import serial
import sys
import numpy as np
import time
import cv2
import pickle
# Create an object handle for serial communication
ser=serial.Serial("/dev/ttyACM0",9600)
# Create an object handle for camera feed
cap=cv2.VideoCapture(1)
# Create an object handle to save the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('out_final_test_track6_wl.avi',fourcc, 20.0, (640,180))

#Class to predict the steering angle
class model_predict:
    def __init__(self):
        ser.write('a'.encode())
        ser.write(9)
        #Load the model
        pkl_filename = "model_real.pkl"
        with open(pkl_filename, 'rb') as file:
            self.model = pickle.load(file)
        
    def open_cam(self):
        
        global cap
        while(True):
            #Read the feed from camera
            _,frame=cap.read()
            height=640
            width=480
            # Scale the input image
            Scale1 = 71
            Scale2 = 71
            #Resize and reshape the image
            frame =frame[300:width,0:height]
            out.write(frame)
            # Resize and reshape the image
            images=cv2.resize(frame,(Scale1,Scale1))
            image=images.reshape(1,Scale1,Scale1,3)
            #Show the ouput image
            cv2.imshow('',images)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            if self.model == "model_real.pkl"
	            prediction=self.model.predict(image)
	            #Predicts the value of the steering angle
	            prediction=(np.argmax(prediction,axis=1))
	            print(prediction)
	            if prediction == 1:
	                prediction = 1
	            elif prediction == 2:
	                prediction = 2
	            if prediction == 3:
	                prediction = 3
	            elif prediction == 4:
	                prediction = 4
	            elif prediction == 5:
	                prediction = 5
	            elif prediction == 6:
	                prediction = 6
	            elif prediction == 7:
	                prediction = 7
	                
	        elif self.model == "model_sim.pkl":
	        	prediction=self.model.predict(image)
	            #Predicts the value of the steering angle
	            prediction=(np.argmax(prediction,axis=1))
	            print(prediction)
	           	if prediction == 0:
	                prediction = 2
	            elif prediction == 1:
	                prediction = 4
	            if prediction == 2:
	                prediction = 6
	            # elif prediction == 4:
	            #     prediction = 4
	            # elif prediction == 5:
	            #     prediction = 5
	            # elif prediction == 6:
	            #     prediction = 6
	            # elif prediction == 7:
	            #     prediction = 7
            
            #Send the predicted value to arduino
            self.arduino(prediction)
      
    
    def arduino(self,value):
        Value=str(value)
        #Send the value to arduino
        Value = Value.encode()
        #Write the values to Motors
        ser.write(Value)
try:
	#Object handle
    handle = model_predict()
    handle.open_cam()
except KeyboardInterrupt:
    pass
finally:
    print("Done")
    ser.write("a".encode())
    cap.release()
    cv2.destroyAllWindows()

