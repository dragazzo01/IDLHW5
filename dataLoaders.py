import torch
import os
import tqdm
from PIL import Image
from torchvision import transforms

class ImageDataset(torch.utils.data.Dataset):
    """Custom dataset for loading image-label pairs."""
    def __init__(self, root, image_size):
        """
        Args:
            root (str): Path to the directory containing the images folder.
            transform (callable): Transform to be applied to the images.
            num_classes (int, optional): Number of classes to keep. If None, keep all classes.
        """
        self.root = root
        self.image_size = image_size
        self.image_paths = []
        #self.labels_file = os.path.join(self.root, "labels.txt")
        self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),  # Converts to [0, 1] range
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizes to [-1, 1] range
            ])
        for subdir, _, files in os.walk(root):
            for file in files:
                self.image_paths.append(os.path.join(subdir, file))

        # self.image_paths = self.image_paths[:15]


    def __len__(self):
        """Returns the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (transformed image, label)
        """
        # Load and transform image on-the-fly
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        #label = self.labels[idx]
        return image#, label
