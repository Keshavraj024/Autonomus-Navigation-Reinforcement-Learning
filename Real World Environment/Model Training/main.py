from threading import Thread
from tqdm import tqdm
import random
import time
import numpy as np
import cv2
import pickle
from car_env import CarEnv
from lane_detection import LaneDetection
from dql_agent import DQLAgent
import tensorflow as tf
import logging

cap =  cv2.VideoCapture(1)
ser = serial.Serial('/dev/ttyACM0',9600)

def main():
    iterations = 50
    epsilon = 1.0 
    epsilon_decay = 1 / 96000 
    ser.write(9)
    min_epsilon = 0.75
    minimum_reward = -200
    stats_every = 10
    agent = DQLAgent()
    env = CarEnv()
    lane = LaneDetection()
    env.run("a")
    logging.info("Automatic mode is active")

    try:
        iteration_rewards = [-200]
        random.seed(1)
        np.random.seed(1)
        tf.set_random_seed(1)
        train_thread = Thread(target=agent.train_in_loop, daemon=True)
        train_thread.start()
        for iteration in tqdm(range(1, iterations + 1), ascii=True, unit="iterations"):
            env.reset()
            if iteration == 50:
                time.sleep(30)
            current_state = env.get_state()
            iteration_reward = 0
            step = 1
            done = False
            while not done:
                slopes = lane.get_slopes(current_state)
                if len(slopes) < 5:
                    if all(i < 0 for i in slopes):
                        slopes.append(-1)
                    elif all(i >= 0 for i in slopes):
                        slopes.append(1)

                if slopes[0] > 0 and slopes[1] > 0 and slopes[2] > 0 and slopes[3] > 0:
                    stop = True

                elif slopes[0] < 0 and slopes[1] < 0 and slopes[2] < 0 and slopes[3] < 0:
                    stop = True

                else:
                    logging.info(agent.get_qs(current_state))
                    action = np.argmax(agent.get_qs(current_state))
                    stop = False

                new_state, reward, done, _ = env.step(action, stop)
                iteration_reward += reward
                agent.update_replay_memory((current_state, action, reward, new_state, done))
                current_state = new_state
                step += 1
                if done:
                    break

            iteration_rewards.append(iteration_reward)
            if not iteration % stats_every or iteration == 1:
                min_reward = min(iteration_rewards[-stats_every:])
                if min_reward > minimum_reward:
                    pkl_filename = "min_model_real.pkl"
                    with open(pkl_filename, 'wb') as file:
                        pickle.dump(agent.model, file)

            if epsilon > min_epsilon:
                epsilon *= epsilon_decay
                epsilon = max(epsilon, min_epsilon)

        agent.end_state = True
        train_thread.join()
        pkl_filename = "model_real.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(agent.model, file)
        logging.info("Saved model to disk")

    finally:
        logging.info('Terminated')


if __name__ == "__main__":
    try:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        main()
    except KeyboardInterrupt:
        pass
    finally:
        logging.info("Done")
        ser.write("a".encode())
        cap.release()
        cv2.destroyAllWindows()
