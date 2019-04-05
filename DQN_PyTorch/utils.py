import numpy as np
from PIL import Image


def _preprocess(img):
    img = Image.fromarray(img)
    img = img.resize((84, 110))
    img = img.crop((0, 26, 84, 110))
    img = img.convert('L')
    img = np.array(img)
    img = img[np.newaxis, np.newaxis, :, :]
    return img
def init_state(img) :
    processed = _preprocess(img)
    state = processed
    for i in range(3) :
        state = np.concatenate((state, processed), axis=1)

    return state

def preprocess(state, next_state) :

    for i in range(3, 0, -1) :
        state[0][i] = state[0][i-1]
    state[0][0] = _preprocess(next_state)

    return state