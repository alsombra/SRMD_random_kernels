import os
import glob
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import numpy as np
from utils import Kernels, load_kernels

TYPES = ('*.png', '*.jpg', '*.jpeg', '*.bmp')

torch.set_default_tensor_type(torch.DoubleTensor)


def Scaling(image):
    return np.array(image) / 255.0


class ImageFolder(data.Dataset):
    """Custom Dataset compatible with prebuilt DataLoader."""
    def __init__(self, root, config=None):
        """Initialize image paths and preprocessing module."""
        self.image_paths = []
        for ext in TYPES:
            self.image_paths.extend(glob.glob(os.path.join(root, ext)))
        self.image_size = config.image_size # LR image size
        self.scale_factor = config.scale_factor
        K, P = load_kernels(file_path='kernels/', scale_factor=self.scale_factor)
        #K = kernels -> K.shape = (15,15,1,358)
        #P = Matriz de projeçao do PCA --> P.shape = (15,225)
        self.randkern = Kernels(K, P)

    def __getitem__(self, index):
        """Read an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if np.array(image).shape==4:
            image = np.array(image)[:,:,:3]
            image = Image.fromarray(image)
        # target (high-resolution image)
        #Random Crop (original method, for any dataset)
        ######################################################
        #transform = transforms.RandomCrop(self.image_size * self.scale_factor) #128x128
        #hr_image = transform(image)
        ######################################################
        
        
       # Face Crop (especific for CelebA)
       # In our case I just want the faces (middle of image). Then I write:
        face_width = face_height = 128 ######## HARDCODED HR.shape = 128 ###############
        j = (image.size[0] - face_width) // 2
        i = (image.size[1] - face_height) // 2
        image = image.crop([j, i, j + face_width, i + face_height])
        hr_image = image

        
        # input (low-resolution image)
        transform = transforms.Compose([
                            transforms.Lambda(lambda x: self.randkern.RandomBlur(x)),
                            transforms.Resize((self.image_size, self.image_size), Image.BICUBIC), 
                            transforms.Lambda(lambda x: Scaling(x)), # LR --> [0,1]
                            transforms.Lambda(lambda x: self.randkern.ConcatDegraInfo(x))
                    ])
        lr_image = transform(hr_image)

        transform = transforms.ToTensor()
        lr_image, hr_image = transform(lr_image), transform(hr_image)

        return image_path, lr_image.to(torch.float64), hr_image.to(torch.float64)

    def __len__(self):
        """Return the total number of image files."""
        return len(self.image_paths)


def get_loader(image_path, config):
    """Create and return Dataloader."""
    dataset = ImageFolder(image_path, config)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers)
    return data_loader
