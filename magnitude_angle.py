import numpy as np
import math

class MagnitudeAngle:
    def __init__(self, gradient_x, gradient_y):
        self.gradient_x = gradient_x
        self.gradient_y = gradient_y

        self._magnitude = None
        self._magnitude_norm = None
        self._angle = None
        self._edge_angle = None
    
    def calculate(self):
        
        self._magnitude, self._magnitude_norm = self.calcuate_magnitude(self._gradient_x, self._gradient_y)
        self._angle, self._edge_angle = self.calculate_angle(self._gradient_x, self._gradient_y)

        return self._magnitude, self._magnitude_norm, self._angle, self._edge_angle

    def calcuate_magnitude(self, gradient_x, gradient_y):
        height, width = gradient_x.shape

        # After gaussing smoothing and gradient computation we have lost a total of 8 rows and 8 columns
        magnitude = np.zeros((height - 8, width - 8))

        # looping over the desired matrix
        for i in range(4,height - 4):
            for j in range(4,width - 4):
                # gradient calculated using root(gx**2 + gy**2)
                temp = (gradient_x[i, j] ** 2) + (gradient_y[i, j] ** 2)
                
                magnitude[i - 4, j - 4] = math.sqrt(temp)

        # Nomralization of Magnitude
        magnitude_norm = magnitude / 360.624
        # same size as original image
        magnitude = np.pad(magnitude, 4, mode='constant')
        magnitude_norm = np.pad(magnitude_norm, 4, mode='constant')

        return magnitude, magnitude_norm
    
    def calculate_angle(self, gradient_x, gradient_y):
        
        height, width = gradient_x.shape
        
        # After gaussing smoothing and gradient computation we have lost a total of 8 rows and 8 columns
        angle = np.zeros((height - 8, width - 8))
        edge_angle = np.zeros((height - 8, width - 8))
        
        # looping over the desired matrix
        for i in range(4,height - 4):
            for j in range(4,width - 4):
                if gradient_x[i, j]  != 0:
                    # gradient angle computed using tan-1(gy/gx)
                    angle[i - 4, j - 4] = math.degrees(math.atan((gradient_y[i, j] / gradient_x[i, j])))
                    edge_angle[i - 4, j - 4] = angle[i - 4, j - 4] + 90
        
        # same size as original image
        angle = np.pad(angle, 4, mode='constant')
        edge_angle = np.pad(angle, 4, mode='constant')

        return angle, edge_angle