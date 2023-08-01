import time
import cv2
import serial

cap =  cv2.VideoCapture(1)
ser = serial.Serial('/dev/ttyACM0',9600)

class CarEnv:

    def __init__(self):
        """Initialize the CarEnv class."""
        self.camera_img = None
        self.Im_width = 71
        self.Im_height = 71
        self.cam_show = True

    def reset(self):
        """Reset the car environment."""
        time.sleep(10)

    def get_state(self):
        """Get the current state of the car environment (frame from the camera)."""
        _, self.frame = cap.read()
        self.frame = self.frame[300:480, 20:580]
        self.frame = cv2.resize(self.frame, (self.Im_width, self.Im_height))
        return self.frame

    def step(self, action, stop):
        """Perform a step in the car environment based on the given action.

        Args:
            action (int): The action to be taken.
            stop (bool): Flag to indicate if the car should stop.

        Returns:
            Tuple: (next_state, reward, done, None)
        """
        done = False
        reward = 1.0

        
        match action:
            case 1: # (Right)
                self.run(1)
            case 2: # (Right)
                self.run(2)
            case 3: # (Right)
                self.run(3)
            case 4: # (straight)
                self.run(4)
            case 5: # (Left)
                self.run(5)
            case 6: # (Left)
                self.run(6)
            case 7: # (Left)
                self.run(7)


        if stop:
            reward = -200
            done = True
            ser.write("9".encode())

        return self.get_state(), reward, done, None

    def run(self, value):
        """Send a command to the car to run in the specified direction.

        Args:
            value (int): The direction value (1, 2, 3, ...).
        """
        Value = str(value)
        Value = Value.encode()
        ser.write(Value)

