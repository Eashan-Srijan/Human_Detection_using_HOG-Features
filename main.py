####################################################
## Developed by: Eashan Kaushik & Srijan Malhotra ##
## Project Start: 27th November 2021              ##
####################################################

from HOG import HOGDescriptor
import matplotlib.pyplot as plt

# main block
if __name__ == '__main__':

    # call HOG class
    hg = HOGDescriptor()
    # Fit on training data
    hg.fit()
    # Transform test data
    hg.test()

    # evaluate the data on test images
    print(f'For Test Images{hg.evaluate()}')
    
    # evaluate the data on train images
    print(f'\n\n\nFor Train Images{hg.evaluate(train=True)}')

    # HOG ASCII Derscriptors
    for name in ['crop001028a.bmp-positive', 'crop001030c.bmp-positive', '00000091a_cut.bmp-negative']:

        with open('Output/' + name.split('-')[0] + '.txt', 'w',encoding='utf-8') as file:
            file.writelines( "%s\n" % item for item in hg.training_discript[name])

    for name in ['crop001278a.bmp-positive', 'crop001500b.bmp-positive', '00000090a_cut.bmp-negative']:
        
        with open('Output/' + name.split('-')[0] + '.txt', 'w',encoding='utf-8') as file:
            file.writelines( "%s\n" % item for item in hg.test_discript[name] )
    
    # Gradient Magnitude of Test Images
    for i in range(0, hg._magnitude_test.shape[0]):
        plt.imsave('Magnitude/' + hg.image_names_test[i], hg._magnitude_test[i], cmap='gray')