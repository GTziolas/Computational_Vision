#2591
from PIL import Image
import sys
from IPython.display import display,Image as jupyter
import numpy as np

#argv[1] is the input photo
image = Image.open(sys.argv[1])
#argv[3] is the threshold
threshold = int(sys.argv[3])

#For the grayscale images
def grayscale_image(array): 
	for i in range(len(array)):
		for j in range(len(array[i])):
			if array[i][j]>threshold:
				array[i][j]=255
			else:
				array[i][j]=0
#For the rgb images                                      
def rgb_image(array):
    for i in range(len(array)):
        for j in range(len(array[i])):
            rgb = array[i][j] 
            red = rgb[0]
            green = rgb[1]
            blue=rgb[2]
            #The average color
            average=((int(red))+(int(green))+(int(blue)))/3
            if average>threshold:
                array[i][j]=255
            else:
                array[i][j]=0
                
#Convert image to array
image_to_array = np.asarray(image)  
img_cp = image_to_array.copy() 
#Get details of the image
print('Shape of image is:',img_cp.shape)
print('The threshold is :',threshold)
print('Doing the work!')  

#Images that have 3 dimensions are rgb and 2 dimensions are grayscale
if (len(img_cp.shape)==3):
    print('Image is rgb')
    rgb_image(img_cp)
else:
    grayscale_image(img_cp)
    print('Image is grayscale')    

#From processed array to image        
final_image=Image.fromarray(img_cp)
#Save it with name argv[2]
final_image.save(sys.argv[2])
final_image.show()
display(jupyter(filename=sys.argv[2]))