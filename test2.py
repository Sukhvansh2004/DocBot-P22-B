import fitz

# Open the image file
pdf_file = fitz.open(r"G:\My Drive\CS671-DL\Hackathon\NayaApp2\NayaApp2 copy\app\testing.pdf")

# Get the first page
page = pdf_file[0]
print(page.get_images()[0])
# Get the image from the page
image = page.get_images()[0]

# Get the image information
image_info = image.get_image_info()

# Check the colormap or color space
colormap = image_info.colorspace

print("Colormap of the image:", colormap)