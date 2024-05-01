import cv2
import fitz
import io
import fitz
from PIL import Image
path = r'G:\My Drive\CS671-DL\Hackathon\NayaApp2\NayaApp2 copy\app\data\file_1_unlocked.pdf'
doc = fitz.open(path, filetype="pdf")
import pprint 
# Load the image
# image = cv2.imread(r'G:\My Drive\CS671-DL\Hackathon\NayaApp2\NayaApp2 copy\app\data\file_1_unlocked.pdf')

# Convert the image to a different color space (e.g., grayscale to colored)
# colored_image = cv2.applyColorMap(image, cv2.IMREAD_REDUCED_GRAYSCALE_2)
# colored_image2 = cv2.applyColorMap(colored_image, cv2.COLOR_RGBA2GRAY)
# # Display the colored image
# cv2.imshow('Colored Image', colored_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

for page in doc:

    # page.get_images(True)
    imglist = page.get_images()[0]
    xref = imglist[0]
    print(xref)
    
    
    
    pix19 = fitz.Pixmap(doc, 19)
    mask = fitz.Pixmap(doc, 25)
    pix = fitz.Pixmap(pix19, mask)
    pix.save(r"G:\My Drive\CS671-DL\Hackathon\NayaApp2\NayaApp2 copy\app\test.png")
    # Save the colored image
    # cv2.imwrite(r'G:\My Drive\CS671-DL\Hackathon\NayaApp2\NayaApp2 copy\app\colored_image.jpg', colored_image)

    # Convert the image to a different color space (e.g., grayscale to colored)