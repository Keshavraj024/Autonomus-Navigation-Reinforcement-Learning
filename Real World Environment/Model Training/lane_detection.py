import cv2
import numpy as np

class LaneDetection:
    """Class representing the lane detection using computer vision."""

    def __init__(self):
        self.slopes = []

    def grayscale(self, img: np.ndarray) -> np.ndarray:
        """Convert image to grayscale.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Grayscale image.
        """
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def hough_lines(self, img: np.ndarray, rho: float, theta: float, threshold: int, min_line_len: int, max_line_gap: int) -> list:
        """Detect lines from the image using Hough transform.

        Args:
            img (np.ndarray): Input image.
            rho (float): Distance resolution of the accumulator in pixels.
            theta (float): Angle resolution of the accumulator in radians.
            threshold (int): Accumulator threshold parameter.
            min_line_len (int): Minimum length of a line (in pixels).
            max_line_gap (int): Maximum allowed gap between points on the same line (in pixels).

        Returns:
            list: List of slopes of detected lines.
        """
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        slopes = [(y1 - y2) / (x1 - x2) for line in lines for x1, y1, x2, y2 in line]
        return slopes

    def process_image(self, image: np.ndarray) -> list:
        """Retrieve the slopes using canny edge detection.

        Args:
            image (np.ndarray): Input image.

        Returns:
            list: List of slopes.
        """
        canny = cv2.Canny(self.grayscale(image), 50, 120)
        slopes = self.hough_lines(canny, 1, np.pi / 180, 10, 10, 250)
        return slopes

    def get_slopes(self, frame: np.ndarray) -> list:
        """Get the slopes from the frame.

        Args:
            frame (np.ndarray): Input frame.

        Returns:
            list: List of slopes.
        """
        frame = cv2.resize(frame, (70, 30))
        slopes = self.process_image(frame)
        return slopes
