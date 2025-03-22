import numpy as np
from PIL import Image
from tqdm import tqdm

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data = unpickle("/ocean/projects/cis250019p/dragazzo/HW5/IDLHW5/data/cifar-10-batches-py/data_batch_4")
print(data.keys())

for i, image_array in tqdm(enumerate(data[b'data']), ascii=True):
    red_channel = image_array[:1024].reshape(32, 32)
    green_channel = image_array[1024:2048].reshape(32, 32)
    blue_channel = image_array[2048:].reshape(32, 32)

    rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)
    image = Image.fromarray(rgb_image, 'RGB')
    image.save(f'/ocean/projects/cis250019p/dragazzo/HW5/IDLHW5/data/cifar-10-batches-py/batch_4/train_img_{i}.png')