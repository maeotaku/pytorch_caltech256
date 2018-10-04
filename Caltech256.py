from __future__ import print_function
from PIL import Image
import os
import os.path
import glob
import sys

from torchvision.datasets.utils import download_url, check_integrity
from torchvision import datasets

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS) and os.path.getsize(filename) > 0

class Caltech256(datasets.ImageFolder):
    base_folder = '256_ObjectCategories'
    url = "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar"
    filename = "256_ObjectCategories.tar"
    tgz_md5 = '67b4f42ca05d46448c6bb8ecd2220f6d'

    def __init__(self, root, train=True,
               transform=None, target_transform=None,
               download=False):

        if download:
            self.download(root)

        if not self._check_integrity(root):
            raise RuntimeError( 'Dataset not found or corrupted.' +
                                ' You can use download=True to download it')
        self.root = os.path.join(root, self.base_folder)
        super(datasets.ImageFolder, self).__init__(root=self.root, transform=transform, loader=default_loader, extensions=IMG_EXTENSIONS)

    def _check_integrity(self, root):
        fpath = os.path.join(root, self.filename)
        if not check_integrity(fpath, self.tgz_md5):
            return False
        return True

    def download(self, root):
        import tarfile
        download_url(self.url, root, self.filename, self.tgz_md5)
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
