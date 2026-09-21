
# Common Libraries
import numpy as np
from PIL import ImageColor

# Functions
def rgb_to_hex(rgb):
    return "#" + '%02x%02x%02x' % rgb

def hex_to_rgb(hex_):
    return np.array(ImageColor.getcolor(hex_, "RGB"))
