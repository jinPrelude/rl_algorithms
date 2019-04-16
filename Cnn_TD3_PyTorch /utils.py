import numpy as np
from PIL import Image
import torch


def init_state(img) :
    state = preprocess(img)
    state = state[np.newaxis, :, :]
    history = np.stack((state, state, state, state), axis=1)

    return history

def preprocess(img) :
    img = Image.fromarray(img)
    img = img.resize((64, 64))
    img = img.convert('L')
    img = np.array(img) / 255
    img = np.float16(img)

    return img

def carRace_output_to_action(output) :
    action = output[0].copy()
    action[1] = (action[1] * 0.5) + 0.5
    action[2] = (action[2] * 0.5) + 0.5
    return action

def carRace_action_to_output(action) :
    output = action.copy()
    output[1] = 2*(output[1] - 0.5)
    output[2] = 2*(output[2] - 0.5)
    output = output[np.newaxis, :]
    output = torch.from_numpy(output)
    return output

def show_state(state) :
        img = Image.fromarray(state[0][3])
        img.show()