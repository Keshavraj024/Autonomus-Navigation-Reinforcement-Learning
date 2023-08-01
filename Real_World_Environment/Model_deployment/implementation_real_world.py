"""
Program to implement the trained model in the Real-World Environment.
"""

import serial
import cv2
import pickle
from typing import Any


class ModelPredict:
    """
    Class to load the trained model and make predictions in a real-world environment.
    """

    def __init__(self) -> None:
        self.ser = serial.Serial("/dev/ttyACM0", 9600)
        self.cap = cv2.VideoCapture(1)
        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.out = cv2.VideoWriter(
            "final_test_track.avi", self.fourcc, 20.0, (640, 180)
        )
        self.model = self.load_model()

    def load_model(self) -> Any:
        """
        Load the trained model from a pickle file.
        """
        self.ser.write(b"a")
        self.ser.write(b"9")
        pkl_filename = "model_real.pkl"
        with open(pkl_filename, "rb") as file:
            model = pickle.load(file)
        return model

    def open_cam(self) -> None:
        """
        Open the camera and make predictions using the trained model.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            height = 640
            width = 480
            Scale1 = 71
            frame = frame[300:width, 0:height]
            self.out.write(frame)
            images = cv2.resize(frame, (Scale1, Scale1))
            image = images.reshape(1, Scale1, Scale1, 3)
            cv2.imshow("", images)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
            prediction = self.model.predict(image)
            self.arduino(prediction)

    def arduino(self, value: Any) -> None:
        """
        Send the prediction value to Arduino.
        """
        value_str = str(value)
        value_bytes = value_str.encode()
        self.ser.write(value_bytes)


try:
    handle = ModelPredict()
    handle.open_cam()
except KeyboardInterrupt:
    pass
finally:
    handle.ser.write(b"a")
    handle.cap.release()
    cv2.destroyAllWindows()
