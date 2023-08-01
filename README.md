# Autonomous Navigation using Deep-Q-Reinforcement Learning

This is a program to train a RL agent in a real and simulation-world environment using Deep Q-Learning and Computer Vision for Lane Detection.

## Dependencies

- Python 3.x
- OpenCV
- Keras
- TensorFlow
- Numpy
- tqdm
- KeyboardInterrupt

## Installation

1. Clone the repository: `git clone https://github.com/Keshavraj024/Autonomus-Navigation-Reinforcement-Learning.git`
2. Install the required dependencies: `pip3 install -r requirements.txt`

## Usage

1. Connect your camera and Arduino to your system.
2. Run the `main.py` file to start the RL agent and lane detection.

## Classes

### CarEnv

The `CarEnv` class represents the real-world environment for the RL agent. It provides functions to reset the environment, get the current state, and take a step based on the agent's action.

### LaneDetection

The `LaneDetection` class handles lane detection using Computer Vision techniques. It provides functions for grayscale conversion and Hough line detection.

### DQLAgent

The `DQLAgent` class is the RL agent itself. It implements the Deep Q-Learning algorithm and trains the neural network model.

## Configuration

- `Memory_fraction`: A float value representing the fraction of memory to be used.
- `iterations`: Number of training iterations.
- `epsilon`: Initial epsilon value for exploration.
- `epsilon_decay`: Epsilon decay rate.
- `min_epsilon`: Minimum epsilon value.
- `minimum_reward`: Minimum reward value.
- `stats_every`: Frequency of displaying training statistics.
