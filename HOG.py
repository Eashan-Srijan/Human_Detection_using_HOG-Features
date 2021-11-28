# The Convolution module is developed by us
from convolution import SeConvolve
from magnitude_angle import MagnitudeAngle
import numpy as np

TRAIN_PATH = './train'
TEST_PATH = './test'

class HOGDescriptor:
    def __init__(self):
        self.train_images = list()

        # Gradient x and y operations Prewitt's Operators
        self._convolution_matrix_gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        self._convolution_matrix_gy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        # Output of step 2
        self._gradient_x = None
        self._gradient_y = None
        self._gradient_x_norm = None
        self._gradient_y_norm = None
        # Output of step 3
        self._magnitude = None
        self._magnitude_norm = None
        # Angle Output
        self._angle = None
        self._edge_angle = None
    
    def to_grayscale(self):
        pass

    def gradient_operation(self):
        # Convolution done on the image_matrix w.r.t gradient x
        gradient_x = SeConvolve(self._smoothed_image, self._convolution_matrix_gx, mode='gradient')
        self._gradient_x, self._gradient_x_norm = gradient_x.convolution()
        
        # Convolution done on the image_matrix w.r.t gradient y
        gradient_y = SeConvolve(self._smoothed_image, self._convolution_matrix_gy, mode='gradient')
        self._gradient_y, self._gradient_y_norm = gradient_y.convolution()
        
        # We compute gradient magnitude, gradient angle and edge angle
        magnitude_angle = MagnitudeAngle(self._gradient_x, self._gradient_y)

        self._magnitude, self._magnitude_norm, self._angle, self._edge_angle = magnitude_angle.calculate()

    def train(self):
        pass

    def test(self):
        pass