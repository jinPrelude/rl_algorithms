import numpy as np
from PIL import Image


def preprocess(img):
    img = Image.fromarray(img)
    img = img.resize((84, 110))
    img = img.crop((0, 26, 84, 110))
    img = img.convert('L')
    img = img.resize((64, 64))
    img = np.array(img) / 255
    img = np.float16(img)
    return img
def init_state(img) :
    state = preprocess(img)
    state = state[np.newaxis, :, :]
    history = np.stack((state, state, state, state), axis=1)

    return history


