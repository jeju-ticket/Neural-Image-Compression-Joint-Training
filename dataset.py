import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import random
from torchvision.transforms import transforms, functional


class ImageFolderVimeo(Dataset):
    def __init__(self, root, transform=None, split="train"):
        from tqdm import tqdm
        self.mode = split
        self.transform = transform
        self.samples = []
        split_dir = Path(root) / Path('vimeo_septuplet/sequences')
        for sub_f in tqdm(split_dir.iterdir()):
            if sub_f.is_dir():
                for sub_sub_f in Path(sub_f).iterdir():
                    self.samples += list(sub_sub_f.iterdir())

        if not split_dir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


class OpenImageDataset(Dataset):
    def __init__(self, root, transform=None, split="openImage"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        # Openimage
        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        self.transform = transform
        self.mode = split

    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert("RGB")
        img = self.resize_crop(img, (256, 256))

        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def resize_crop(img, patch_size):
        patch_x, patch_y = patch_size
        width, height = img.size

        random_resize_factor = 0.3  # random.random() * 0.1 + 0.3  #
        crop_size = [round(patch_x / random_resize_factor), round(patch_y / random_resize_factor)]

        random_crop_x1 = 0 + int(random.random() * (width - crop_size[1] - 2))
        random_crop_y1 = 0 + int(random.random() * (height - crop_size[0] - 2))
        random_crop_x2 = random_crop_x1 + crop_size[1]
        random_crop_y2 = random_crop_y1 + crop_size[0]

        random_box = (random_crop_x1, random_crop_y1, random_crop_x2, random_crop_y2)
        randomCropPatch = img.crop(random_box)

        randomCropPatch = randomCropPatch.resize((patch_x, patch_y), Image.BICUBIC)

        return randomCropPatch


class Kodak24Dataset(Dataset):
    def __init__(self, root, transform=None, split="kodak24"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        self.transform = transform
        self.mode = split

    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


class VideoFolder(Dataset):
    def __init__(self, root, mode='train', video_name=None, transform=None, random_two_frames=False, num_frames=None):
        from tqdm import tqdm
        self.mode = mode
        self.transform = transform
        if self.mode == 'UVG':
            # UVG
            root_dir = Path(root) / Path('UVG/videos')
            self.samples = []
            self.video_names = []
            for sub_f in tqdm(root_dir.iterdir()):
                if sub_f.name == video_name:
                    if num_frames is None:
                        each_videos = [img for i, img in enumerate(sorted(Path(sub_f).iterdir())) if img.is_file()]
                    else:
                        each_videos = [img for i, img in enumerate(sorted(Path(sub_f).iterdir())) if
                                       img.is_file() and i < num_frames]
                    self.samples += each_videos
                    self.video_names += [Path(sub_f).name] * len(each_videos)

        elif mode == 'train':
            root_dir = Path(root) / Path('vimeo_septuplet/sequences')
            self.samples = []
            for i, sub_f in tqdm(enumerate(root_dir.iterdir())):
                if sub_f.is_dir() and i > 0:
                    for sub_sub_f in Path(sub_f).iterdir():
                        each_videos = sorted(list(sub_sub_f.iterdir()))
                        if random_two_frames:
                            each_videos = random.sample(each_videos, 2)
                        self.samples.append(each_videos)
            if not root_dir.is_dir():
                raise RuntimeError(f'Invalid directory "{root}"')

        else:
            root_dir = Path(root) / Path('vimeo_septuplet/sequences')
            self.samples = []
            for i, sub_f in tqdm(enumerate(root_dir.iterdir())):
                if sub_f.is_dir() and i <= 0:
                    for sub_sub_f in Path(sub_f).iterdir():
                        each_videos = sorted(list(sub_sub_f.iterdir()))
                        if random_two_frames:
                            each_videos = random.sample(each_videos, 2)
                        self.samples.append(each_videos)
            if not root_dir.is_dir():
                raise RuntimeError(f'Invalid directory "{root}"')

    def __getitem__(self, index):
        if self.mode == 'UVG':
            video_name = self.video_names[index]
            img = self.samples[index]
            if self.transform:
                return video_name, self.transform(Image.open(img).convert("RGB"))
            else:
                return img
        elif self.mode == 'train':
            imgs = self.samples[index]
            imgs = [Image.open(img).convert("RGB") for img in imgs]

            # RandomCrop for a sequence
            i, j, h, w = transforms.RandomCrop.get_params(imgs[0], output_size=(256, 256))
            p1, p2 = torch.rand(1), torch.rand(1)

            # Random Flipping for a sequence
            for i in range(len(imgs)):
                imgs[i] = functional.crop(imgs[i], i, j, h, w)
                if p1 < 0.5:
                    imgs[i] = functional.vflip(imgs[i])
                if p2 < 0.5:
                    imgs[i] = functional.hflip(imgs[i])

            if self.transform:
                return [self.transform(img) for img in imgs]
            else:
                return imgs

        else:  # validation
            imgs = self.samples[index]
            imgs = [Image.open(img).convert("RGB") for img in imgs]
            if self.transform:
                return [self.transform(img) for img in imgs]
            else:
                return imgs

    def __len__(self):
        return len(self.samples)
