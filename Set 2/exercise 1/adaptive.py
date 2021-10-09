#2591

import sys
import numpy as np
from PIL import Image
from math import ceil

#argv[1] is the input photo
#argv[2] is the output photo
#argv[3] is the window size
input_image = Image.open(sys.argv[1])
out_image = sys.argv[2]
window_size = int(sys.argv[3])
#image as numpy array
image_array = np.asarray(input_image)
image_array_cp = image_array.copy()
images_cropped=[]
images=[]

width, height = input_image.size
print('Dimensions of the image (height,width):', image_array.shape)

#Decide if greyscale or rgb array as input
def is_greyscale_array(array):
    if(len(array.shape) == 2):
        return True
    else:
        return False

def ypologise_antikeimeniki_otsu(A, k):
    pixels_tmima1 = A[A < k]
    pixels_tmima2 = A[A >=k]
    mu1 = np.mean(pixels_tmima1)
    mu2 = np.mean(pixels_tmima2)
    mu_synoliko = np.mean(A.flatten())
    pi1 = len(pixels_tmima1) / (len(pixels_tmima1) + len(pixels_tmima2))
    pi2 = len(pixels_tmima2) / (len(pixels_tmima1) + len(pixels_tmima2))
    antikeimeniki_synartisi = pi1 * (mu1 - mu_synoliko)**2 + pi2 * (mu2 - mu_synoliko)**2
    return(antikeimeniki_synartisi)
    
def otsu_thresholder(image):
    kalytero_katwfli = 0
    kalyterh_timi = 0
    for i in range(1, 256):
        obj_otsu = ypologise_antikeimeniki_otsu(cropped_array, i)
        if(obj_otsu > kalyterh_timi):
            kalytero_katwfli = i 
            kalyterh_timi = obj_otsu
    if (is_greyscale_array(cropped_array)):
        res = katwfliwsh_eikonas(image, kalytero_katwfli)
    else:
        res = katwfliwsh_eikonas_rgb(image, kalytero_katwfli)        

    return(res)

#greyscale
def katwfliwsh_eikonas(image, threshold):
    res = np.zeros_like(image)
    res[image < threshold] = 0
    res[image >=threshold] = 255
    return( np.uint8(res) )

#rgb
def katwfliwsh_eikonas_rgb(image, threshold):
    for i in range(len(image)):
        for j in range(len(image[i])):
            rgb = image[i][j] 
            red = rgb[0]
            green = rgb[1]
            blue=rgb[2]
            #The average color
            average=((int(red))+(int(green))+(int(blue)))/3
            if average>threshold:
                image[i][j]=255
                return (np.uint8(image))
            else:
                image[i][j]=0
                return (np.uint8(image))
            
i_increment = ceil(width/window_size)
j_increment = ceil(height/window_size)

#Crop image into smaller ones, apply otsus thresholding
#And rebuild the original image from the crops
for j in range(0,j_increment):
     for i in range(0,i_increment):
         wi = window_size*i
         wj = window_size*j
         wi_plus = wi + window_size
         wj_plus = wj + window_size  
         #bigger than the whole image
         if (wi_plus > width):
             wi_plus = width
         if (wj_plus > height):
             wj_plus = height
         #the area to crop           
         area = (wi, wj, wi_plus, wj_plus)
         cropped_img = input_image.crop(area)
         cropped_array = np.array(cropped_img)
         
         A_otsu = otsu_thresholder(cropped_array)
         images_cropped.append(A_otsu)
  
if (is_greyscale_array(cropped_array)): # greyscale
    print("It's greyscale")
    for x in images_cropped:
        images.append(Image.fromarray(x,'L'))
    new_im = Image.new('L', (width,height))    
else: #rgb
    print("It's RGB")
    for x in images_cropped:
        images.append(Image.fromarray(x,'RGBA'))
    new_im = Image.new('L', (width,height))    

#Rebuild the final image
i=0
for x in range(0,j_increment):
    for y in range(0,i_increment):
        new_im.paste(images[i], (y*window_size, x*window_size))
        i+=1

#Show and save
new_im.show()      
new_im.save(out_image)       
           