import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('test_image.jpg')

def make_coordinates(image,line_parameter):
    slope, intercept = line_parameter
    y1 = image.shape[0]
    y2 = int(y1*(3/5))  
    # Note : y = mx+b
    # x = (y-b)/m
    x1 =int((y1-intercept)/slope)
    x2 =int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])


def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    for line in lines: # each line is 2d space with 1R-4C
            x1,y1,x2,y2 = line.reshape(4) # put each line 1 D array
            #create 1 dimension polynomial with this class
            parameters = np.polyfit((x1,x2),(y1,y2), 1) # 1 degree polynomial
            # Y= mx + c
            slope = parameters[0]
            intercept = parameters[1]
#  Note : lane will be like this ->   / \  
# / -> x increase y decrease -> its in numpy
            if slope < 0:
                left_fit.append((slope,intercept))
            else:
                 right_fit.append((slope,intercept))
              # calculate avarge 
              
    left_fit_average = np.average(left_fit,axis=0) # axis ->0 calculate vertically coumn wise
    right_fit_average = np.average(right_fit,axis=0)
    left_line = make_coordinates(image, left_fit_average) 
    right_line = make_coordinates(image, right_fit_average) 
    return np.array([left_line,right_line])            
                 
def display_lines(image,lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines: # each line is 2d space with 1R-4C
            x1,y1,x2,y2 = line.reshape(4) # put each line 1 D array
            cv2.line(lines_image, (x1,y1),(x2,y2),(255,0,0),10)
    return lines_image;

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

# In road their are right lane and left lane which are sperated by dvider
# we need a traingle which will suggest |R : L| => [: l]   
def region_of_interest(image):
    height = image.shape[0] # Y axix -> 0
    #triangle = np.array([(200,height),(1100,height),(550,250)])
    
    #Note 
    polygon = np.array([[(200,height),(1100,height),(550,250)]])

    mask = np.zeros_like(image) # black image of image size
    # fill the mask with the given triangle/polynomial with white
    cv2.fillPoly(mask,polygon, 255)
    # fillPoly take several polygon i,e array of array  [[(200,height),(1100,height),(550,250)]]
    
    # Now this mask image can be used in original image to get the area of concerned
    # We can use bitwise & operator to do so
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
    
    
    # Y = mx + b is the equation of lines
    # On X-Y Plot for any given point infinite numbers of line can be drawn, which will have distinct m.n values as X,Y will be constant points
    # If we plot m-n axis for all the lines on given points it will draw a line on M-N axis, this is called parametric space
    
    
    
# creating a copy of an image using numpy 
# image is mutable variablexxxxxx

lane_image = np.copy(image)
canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]),minLineLength=50, maxLineGap=5)
# 2 precisin
# np.pi/180 is theta
# 100 - threhold criteria in bin
# np.array([]) - jsut a palce holder
# minLineLength = 40 any line with less than 40 PX are rejected
#  maxLineGap  This indicates the maximum distance in pixels between segmented lines which we will allow to be connected

averaged_lines = average_slope_intercept(lane_image, lines)

line_image = display_lines(lane_image,averaged_lines)

# Add the lines
# lane_image and line_image have same dimension and hence mergin this 2 image, original wit intensity 0.8 and lines image 1 so that line will be clearly visibale in 
# the lane_image
combo_image = cv2.addWeighted(lane_image,0.8, line_image, 1, 1)

 
cv2.imshow("canny", combo_image)
#plt.imshow(canny)
#plt.show()
cv2.waitKey(0)