from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px, pandas as pd, datetime, numpy as np
import argparse, cv2
import base64
from PIL import Image
from io import BytesIO

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')
app = Dash(__name__)#, external_stylesheets=external_stylesheets, title='Otsu project', update_title='Loading...')

img = "chameleon.jpg"

from dash import Dash, html
import base64
from PIL import Image


#Using direct image file path
image_path = 'chameleon.jpg'

#Using Pillow to read the the image
pil_img = Image.open("chameleon.jpg")

# Using base64 encoding and decoding
def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')

#img = cv2.imread("chameleon.jpg")
# img_list = [img1, img2, img3]

# color_space_list = {'rgb': "cv2.COLOR_BGR2HSV",
                    # 'hsv': "cv2.COLOR_BGR2HSV",
                    # 'lab': ""}

def convert_color_space_arr(img_src, conversion_code):
    # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", type=str, default=img_src,
    #     help="path to input image")
    # args = vars(ap.parse_args())

    # load the original image
    image = cv2.imread(img_src)
    # image_after = cv2.cvtColor(image, conversion_code)
    return image

def numpyndarray_to_src(imp_src, conversion_code):
    arr = convert_color_space_arr(imp_src, conversion_code)
    # Convert NumPy array to PIL Image
    img_pil = Image.fromarray(arr)

    # Create an in-memory binary stream (BytesIO object)
    img_byte_array = BytesIO()

    # Save the image to the in-memory stream as PNG format
    img_pil.save(img_byte_array, format='PNG')

    # Convert the binary stream to a base64 encoded string
    img_base64 = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')

    # Construct the data URI
    img_src = f"data:image/png;base64,{img_base64}"

    return img_src

# def convert_color_space(img_src, conversion_code):
#     arr = convert_color_space_arr(img_src, conversion_code)
#     numpyndarray_to_src(arr)

# # def convert_color_space_multiple(img_list, conversion_code):
# #     converted_list=[]
# #     for i in img_list:
# #         converted_list.append(convert_color_space(i, conversion_code))
# #     return converted_list

# # def set_channel(img):
# #     # Create a blank image with the same size as the original image
# #     red_img = np.zeros_like(img)

# #     # Set only the red channel to the corresponding channel values from the original image
# #     red_img[:,:,2] = img[:,:,2]  # Red channel

# #     result = numpyndarray_to_src(red_img)
# #     return result

# converted_img = convert_color_space(img, cv2.COLOR_BGR2HSV)
#converted_img = set_channel(img)

# img_src_list = [converted_img, converted_img_2, converted_img_3]
# img_list = [html.Img(src=img_src, style={'width': '70%'}) for img_src in img_src_list]

image_path = 'chameleon.jpg'

app.layout = html.Div([
    html.H1('Dash Puppies'),
    html.Img(id="display_image", style={'width': '70%'})

])


@callback(
    Output(component_id='display_image', component_property='src'),
    Input(component_id='display_image', component_property='src'),
)
def display_image():
    image = numpyndarray_to_src(image_path, None)
    return image


if __name__ == '__main__':
    app.run(debug=True)
