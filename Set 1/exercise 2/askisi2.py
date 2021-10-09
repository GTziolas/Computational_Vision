#2591
from PIL import Image
import sys
import numpy as np
from IPython.display import display,Image as jupyter

def transform(array, a1, a2, a3, a4, a5, a6):    
    rows = array.shape[0]
    cols = array.shape[1]
    #This new array will be the representation of the new picture
    new_array = np.zeros((rows,cols))
    
    for r in range(0,rows):
        for c in range(0,cols):
            x = r-(rows/2)
            y= c-(cols/2)
            xx = (a1*x)+(a2*y)+a3 
            yy = (a4*x)+(a5*y)+a6
            rounded_new_rows = round(xx + (rows/2))
            rounded_new_cols = round(yy + (cols/2))
            
            if(rounded_new_rows>=0 and rounded_new_rows<rows and rounded_new_cols>=0 and rounded_new_cols<cols):
                new_array[r][c]=array[rounded_new_rows][rounded_new_cols]
    return new_array
#input image argv[1]
image_array = np.array(Image.open(sys.argv[1]))

#inputs: argv[3] to argv[8] = a1 to a6
a1 = float(sys.argv[3])
a2 = float(sys.argv[4])
a3 = float(sys.argv[5])
a4 = float(sys.argv[6])
a5 = float(sys.argv[7])
a6 = float(sys.argv[8])

#Execute transform
new_array = transform(image_array, a1, a2, a3, a4, a5, a6)
#Create final image from modified array
final_image = Image.fromarray(new_array)
display(jupyter(filename=sys.argv[2]))
#Convert to png writable 
final_image = final_image.convert("L")
#argv[2] is the output file
final_image.save(sys.argv[2])
