import cv2
import numpy as np

image = cv2.imread('test_image.jpg')




def canny(image):
    # Current image is RGB (3 Dimension [0-255] for each of RGB)
    # We can convert this into greyscale image for easy calcualation
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Remove noise , we will use gaussianBlur , with kernel (5,5)
    
    blur = cv2.GaussianBlur(gray, (5,5),0)
    
    #This we look for edge detectation
    
    #Gradient : change in brightness (0->255)
    #[0 , 0 , 255 , 255]
    #Black : 0
    #white: 255 
    
    # f(x,y): what opeartor can be used to see the rapid change in color
    
    # derivative (f(x,y))
    
    
    # OpenCv canny function provides the way to calulate gradient
    
    
    canny = cv2.Canny(blur, 50,150) # 50 and 150 are low and high threshold which are in ration of 1:3 as per documentation
    return canny


# creating a copy of an image using numpy 
# image is mutable variable

lane_image = np.copy(image)
cv2.imshow("canny", canny(lane_image))
cv2.waitKey(0)
