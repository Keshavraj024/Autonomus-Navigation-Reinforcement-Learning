"""
Program to train the RL-agent in the Real world environment using 
Deep Q-Learning and Computer vision for Lane detection
"""
#Import the required modules
import glob
import os
import sys
import random
from tqdm import tqdm
import math
import time
import numpy as np
import cv2
import logging
from collections import deque
import serial
from keyboardinterrupt import KBHit
from keras.models import model_from_json
import pickle
from keras.applications.xception import Xception
from keras.models import Sequential, Model,load_model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Conv2D, AveragePooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
import keras.backend as backend
from threading import Thread
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#Create object handle for Camera
cap =  cv2.VideoCapture(1)
#Create Object handle for serial communication
ser = serial.Serial('/dev/ttyACM0',9600)

#Real-world environment class
class carenv:

    def __init__(self):
        self.camera_img = None
        self.Im_width =71
        self.Im_height =71
        self.cam_show = True

 
    def reset(self):
        time.sleep(10)
        
    def get_state(self):
        #Read the frame
        _,self.frame = cap.read()
        #Resize the frame
        self.frame =self.frame[300:480,20:580]
        self.frame = cv2.resize(self.frame,(self.Im_width,self.Im_height))
        #Show the frame 
        # cv2.imshow('',self.frame)
        # cv2.waitKey(10)
        return self.frame
   
    
    def step(self,action,start_time,stop):
        #Step function to obtain next state,action and reward
        done = False
        reward = 1.0
        #print(action)
        # if action == 0:#(Right)
        #     self.run(0)
            #reward = 0
        if action == 1:#(Right)
            self.run(1)
            #reward = 0
        elif action == 2:#(Right)
            self.run(2)
            #reward = 0
        elif action == 3:#(Right)
            self.run(3)
            #reward = 0
        elif action == 4:#(straight)
            self.run(4)
            #reward = 0
        elif action == 5:#(Left)
            self.run(5)
            #reward = 0
        elif action == 6:#(Left)
            self.run(6)
            #reward = 0
        elif action == 7:#(Left)
            self.run(7)

        if stop:
            reward = -200
            done = True
            ser.write("9".encode())
                
        return self.get_state(),reward,done,None

    def run(self,value):
        Value = str(value)
        Value = Value.encode()
        ser.write(Value)

# Lane detection class using Computer Vision
class lanedetection:
    
    def __init__(self):
        self.slopes = []
        
    def grayscale(self,img):
        #Convert image to grayscale
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def hough_lines(self,img, rho, theta, threshold, min_line_len, max_line_gap):
        # Detect the lines from the image
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        # Create a empty list
        left_slope = []
        right_slope = []
        slopes = []
        for line in lines:
            
            x1,y1,x2,y2 = line.reshape(4)
            
            slope = (y1-y2)/(x1-x2)
            slopes.append(slope)
        return slopes


    def processImage(self,image):
        #Function to retrieve the canny edge detection
        canny = cv2.Canny(self.grayscale(image), 50, 120)
        slopes = self.hough_lines(canny, 1, np.pi/180, 10, 10,250)
        return  slopes


    def main(self,frame):
        #Function to return the line slopes
        frame = cv2.resize(frame,(70,30))
        slopes = self.processImage(frame)
        return slopes

# RL-agent class
class DQLAgent:

    def __init__(self):
        #Discount factor to discount future rewards
        self.Discount = 0.99
        self.sample_batch_size = 16
        self.update_target = 5
        self.prediction_batch_size = 1
        self.training_batch_size = self.sample_batch_size // 4
        #Replay memory size
        self.replay_memory_size = 100
        self.min_replay_memory_size = 50
        self.replay_memory = deque(maxlen = self.replay_memory_size)
        #Dimension of the input image
        self.im_height =71
        self.im_width = 71
        self.model = self.model_created()
        self.target_model = self.model_created()
        self.target_model.set_weights(self.model.get_weights())
        self.graph = tf.get_default_graph()
        self.end_state = False
        self.last_logged_episode = 0
        self.initialise_training = False
        self.target_update_counter = 0
        

    def model_created(self):
        
        #Neural network model for model training
        model.add(Conv2D(128, (3, 3), input_shape=(self.im_height,self.im_width,3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Flatten())
        model.add(Dense(1024,activation = "relu"))
        model.add(Dense(512,activation = "relu"))
        model.add(Dense(7,activation='linear'))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        #Fine-tuning of the model
        # pkl_filename = "pickle_mod_71_71_a.pkl"
        # with open(pkl_filename, 'rb') as file:
        #     model = pickle.load(file)
        #     model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        return model
        

    def update_replay_memory(self,transistion):
        #Function to update the replay memory buffer
        self.replay_memory.append(transistion)

    def train(self):
        #Function to train the model
        x = []
        y = []
        if len(self.replay_memory) < self.min_replay_memory_size:
            return
        #Sample the training data 
        sample_batch = random.sample(self.replay_memory,self.sample_batch_size)
        #Get the sample of current states
        for transistion in sample_batch:
            current_states = np.array([transistion[0]])/255.0
        with self.graph.as_default():
            #Model prediction of Q-Values
            current_qlist = self.model.predict(current_states,self.prediction_batch_size)
        #Get the sample of current states
        for transistion in sample_batch:
            future_states = np.array([transistion[3]])/255.0
        with self.graph.as_default():
            #Model prediction of current Q-Values
            future_qlist = self.target_model.predict(future_states , self.prediction_batch_size)
        for index , (current_state,action,reward,new_state,done) in enumerate(sample_batch):
            if not done:
                q_max = np.max(future_qlist[index])
                q_new = reward + self.Discount * q_max
            else:
                q_new = reward

            target_qlists = current_qlist[index]
            target_qlists[action] = q_new
            #Append the training features and labels
            x.append(current_state)
            y.append(target_qlists)
        with self.graph.as_default():
            #Train the model
            self.model.fit(np.array(x)/255.0 , np.array(y),batch_size = self.training_batch_size,verbose = 0,shuffle = False)
        #update the target model weights after n number of episodes
        if self.target_update_counter > self.update_target:
            self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter += 1
    
    def get_qs(self,state):
        # Get the Q-values
        state = (np.array(state).reshape(-1,*state.shape)/255)[0]
        return self.model.predict(state)
    
    def train_in_loop(self):
        #Function to train the model in continuous loop
        with self.graph.as_default():
            self.model.fit(x , y ,verbose = False , batch_size = 1)
        self.initialise_training = True
        while True:
            if self.end_state:
                return
            self.train()
            time.sleep(0.01)

def main():

    Memory_fraction = 1.0
    iterations = 50
    epsilon = 1.0           #Initial epsilon
    epsilon_decay = 1/96000 #Epsilon decay rate
    ser.write(9)  
    min_epsilon = 0.75
    Min_reward = -200
    stats_every = 10
    agent = DQLAgent()
    env = carenv()
    lane = lanedetection()
    env.run("a")
    print("Automatic mode")
    
    try:
        iteration_rewards = [-200]
        random.seed(1)
        np.random.seed(1)
        tf.set_random_seed(1)
        #Start the thread
        train_thread = Thread(target = agent.train_in_loop,daemon = True)
        train_thread.start()
        for iteration in tqdm(range(1,iterations+1),ascii = True , unit = "iterations"):
            #Reset the environment
            env.reset()
            if iteration == 50:
                time.sleep(30)
            #Get the current state
            current_state = env.get_state()
            iteration_reward = 0
            step = 1
            iteration_begin  = time.time()
            done = False
            while not done:
                # Stop condition if the agent crosses the boundary
                slopes= lane.main(current_state)
                if len(slopes) < 5:
                    if all(i < 0 for i in slopes):
                        slopes.append(-1)
                    elif all(i >= 0 for i in slopes):
                        slopes.append(1)

                if slopes[0] > 0 and slopes[1] > 0 and slopes[2]   > 0 and slopes[3]   > 0:
                    stop = True
                    
                elif slopes[0] < 0 and slopes[1] < 0 and slopes [2] < 0 and slopes[3]   < 0:
                    stop = True
                    
                else:
                    print(agent.get_qs(current_state))
                    action = np.argmax(agent.get_qs(current_state))
                    stop = False
                #Get the next state,reward
                new_state,reward,done, _ = env.step(action,start_time,stop)
                iteration_reward += reward
                #Update the replay buffer
                agent.update_replay_memory((current_state,action,reward,new_state,done))
                current_state = new_state
                step += 1
                if done:
                    break

            iteration_rewards.append(iteration_reward)
            if not iteration % stats_every or iteration == 1:
                #Calculate the average,minimum and maximum reward
                avg_reward = np.mean(iteration_rewards[-stats_every:])
                min_reward = min(iteration_rewards[-stats_every:])
                max_reward = max(iteration_rewards[-stats_every:])
      			#Save the model
                if min_reward > Min_reward:
                    pkl_filename = "min_model_real.pkl"
                    with open(pkl_filename, 'wb') as file:
                            pickle.dump(agent.model, file)

            #Update the epsilon value
            if epsilon > min_epsilon:
                epsilon *= epsilon_decay
                epsilon = max(epsilon,min_epsilon)

        agent.end_state = True
        training_thread.join()
        #Save the final model
        pkl_filename = "model_real.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(agent.model, file)
        print("Saved model to disk")
    
    finally:
        print('Terminated')
        
        
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print("Done")
        ser.write("a".encode())
        #Close the cam
        cap.release()
        cv2.destroyAllWindows()
