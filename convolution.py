####################################################
## Developed by: Eashan Kaushik & Srijan Malhotra ##
## Project Start: 29th October 2021               ##
####################################################

import numpy as np

class SeConvolve:

    def __init__(self, image_matrix, kernel, mode='gradient'):
        # input image matrix
        self.image_matrix = image_matrix
        # inpur kernel
        self.kernel = kernel
        # output after convolution with the kerne
        self._output = None
        # normalized output
        self._output_norm = None
        # mode smoothing or gradient
        self.mode = mode

    #######################
    ## Getter and Setter ##
    #######################

    @property
    def output(self):
        return self._output

    @property
    def output_norm(self):
        return self._output_norm

    #######################
    #######################

    def convolution(self):
      
      height, width = self.image_matrix.shape
      
      if self.mode == 'gradient':
        
        # code to find the gradients 
        self._output = np.zeros((height - 2, width - 2))
        
        # looping over the desired matrix
        for i in range(1,height - 1):
          for j in range(1,width - 1):
            # martix multiplication for gradient computation
            # prewit convolution after gaussian smoothing leads to loss of 4 rows and 4 columns thats why we start from i-4 and j-4
            self._output[i - 1,j - 1] = (self.kernel[0, 0] * self.image_matrix[i - 1, j - 1]) + (self.kernel[0, 1] * self.image_matrix[i - 1, j]) + (self.kernel[0, 2] * self.image_matrix[i - 1, j + 1]) + \
                      (self.kernel[1, 0] * self.image_matrix[i, j - 1]) + (self.kernel[1, 1] * self.image_matrix[i, j]) + (self.kernel[1, 2] * self.image_matrix[i, j + 1]) + (self.kernel[2, 0] * self.image_matrix[i + 1, j - 1]) + \
                      (self.kernel[2, 1] * self.image_matrix[i + 1, j]) + (self.kernel[2, 2] * self.image_matrix[i + 1, j + 1])

      # we call the normalize function to normalize the output
      self.normalize()

      self._output = np.pad(self._output, 1, mode='constant')

      return self._output, self._output_norm

    # normalize
    def normalize(self):
      if self.mode == 'gradient':
        # normalize using sum of absolute values
        temp_output = np.absolute(self._output)
        self._output_norm = temp_output / 3
        self._output_norm = np.pad(self._output_norm, 1, mode='constant')
