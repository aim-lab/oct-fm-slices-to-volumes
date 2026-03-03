from torchvision import transforms
import torch

from PIL import ImageDraw, Image
import numpy as np
import random

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_transform(is_train, args):
    normalize = args.model != "mflm"
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if "fundus" not in args.dataset_name.lower():
        if is_train == 'train':
            # this should always dispatch to transforms_imagenet_train
            transform_og = create_transform(
                input_size=args.img_size,
                # scale = (0.3, 1), ###We only crop up to half of the image
                # ratio = (1,1),
                no_aug=args.no_augment,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa if args.aa!= "None" else None,
                interpolation='bicubic',
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                normalize=normalize,
                mean=mean,
                std=std,
            )

            list_transforms = transform_og.transforms
            if args.dataset_name == "rambam":
                # list_transforms = [SplitImage()] + list_transforms
                list_transforms = [SplitImage()] + [RemoveAnnot(), RemoveAnnot((27,482),(48,474))] + list_transforms
            if "hy" in args.dataset_name:
                # list_transforms = [CropBottom()] + list_transforms
                list_transforms = [CenterRetinaTransform()] + list_transforms

            new_transform = transforms.Compose(list_transforms)

            return new_transform

        # eval transform
        t = []
        if args.img_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(args.img_size / crop_pct)
        if args.dataset_name == "rambam":
            t.append(SplitImage())
            t.extend([RemoveAnnot(), RemoveAnnot((27,482),(48,474))])
        if "hy" in args.dataset_name:
            # t.append(CropBottom())
            t.append(CenterRetinaTransform())
        t.append(
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        )
        t.append(transforms.CenterCrop(args.img_size))
        t.append(transforms.ToTensor())
        if normalize:
            t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)

    else:
        ### Transorm for Fundus:
        rem_lower_left = RemoveAnnot((0,940),(290,880))
        rem_lower_right = RemoveAnnot((1850,940),(1914,450))
        rem_upper_right = RemoveAnnot((1750,70),(1914,2))
        rem_upper_left = RemoveAnnot((0,60),(340,2))

        list_transforms = [rem_lower_left, rem_lower_right, rem_upper_right, rem_upper_left]
        list_transforms.append(pad_image_to_square)
        list_transforms.append(transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True))

        if is_train == "train":
            list_transforms.extend([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply(
                            [transforms.ColorJitter(contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.2, 0.2))],
                            p=0.9
                        ),
                    ])
        list_transforms.append(transforms.ToTensor())
        # if normalize:
        #     list_transforms.append(transforms.Normalize(mean, std))
        return transforms.Compose(list_transforms)


class SplitImage(object):
    """
    Transform to get the oct part of the image.

    Args:
        idx_split (int): The column used to split the images.

    Returns:
        Callable: A callable object that splits the input.

    Example:
        transform = SplitImage(496)
        rotated_image = transform(image)
    """

    def __init__(self, idx_split_horiz=496, idx_split_vert=490):
        self.idx_split_horiz = idx_split_horiz
        self.idx_split_vert = idx_split_vert

    def __call__(self, x):
        width, height = x.size
        oct_img = x.crop((self.idx_split_horiz, 0, width, self.idx_split_vert))
        return oct_img
    

class RemoveAnnot(object):
    """
    Remove Textual annotations in the image

    Args:
        lb_corner : (The left bottom index)
        ru_corner : (The right upper index)

    Returns:
        Callable: A callable object that splits the input.

    Example:
        transform = RemoveAnnot((125, 128), (130, 132))
        rotated_image = transform(image)
    """

    def __init__(self, lb_corner=(8,487), ru_corner=(27, 436)):
        self.lb_corner = lb_corner
        self.ru_corner = ru_corner

    def __call__(self, x):
        x1, y1 = self.lb_corner
        x2, y2 = self.ru_corner
        rect_coords = (x1, y2, x2, y1)

        draw = ImageDraw.Draw(x)
        draw.rectangle(rect_coords, fill="black")

        return x


class CenterRetinaTransform:
    def __init__(self, padding_mode="constant", fill_value=0):
        self.padding_mode = padding_mode  # "constant", "edge", or "reflect"
        self.fill_value = fill_value      # Background fill for constant padding

    def __call__(self, img):
        gray = np.array(img.convert("L"))
        img = np.array(img)  # Convert PIL image to numpy array
        h, w = img.shape[:2]
        is_rgb = len(img.shape) == 3  # Check if image is RGB

        # Compute vertical intensity profile
        intensity_profile = np.mean(gray, axis=1)  # Average intensity across width

        # Detect retina region (brightest part)
        threshold = np.percentile(intensity_profile, 90)  # Adaptive threshold
        retina_indices = np.where(intensity_profile > threshold)[0]

        if len(retina_indices) == 0:
            return Image.fromarray(img)  # If no retina is detected, return original

        retina_center = np.mean(retina_indices).astype(int)
        img_center = h // 2

        # Compute vertical shift required
        shift = img_center - retina_center

        # Apply vertical translation using padding
        if shift > 0:  # Move down
            pad_width = ((shift, 0), (0, 0)) if not is_rgb else ((shift, 0), (0, 0), (0, 0))
            padded_img = np.pad(img, pad_width, mode=self.padding_mode, constant_values=self.fill_value)
            aligned_img = padded_img[:h, :]
        else:  # Move up
            pad_width = ((0, -shift), (0, 0)) if not is_rgb else ((0, -shift), (0, 0), (0, 0))
            padded_img = np.pad(img, pad_width, mode=self.padding_mode, constant_values=self.fill_value)
            aligned_img = padded_img[-h:, :]

        return Image.fromarray(aligned_img).convert("RGB")


class CropBottom(object):
    """
    Tranformation to crop the bottom of the
    Args:
        pixels: number of pixel to crop of the bottom

    Returns:
        Callable: A callable object that splits the input.

    Example:
        transform = CropBottom(150)
        cropped_image = transform(image)
    """
     
    def __init__(self, pixels=150):
        self.pixels = pixels
        
    def __call__(self,img):
        width, height = img.size
        return img.crop((0, 0, width, height - self.pixels))

def crop_bottom(img, pixels=150):
    width, height = img.size
    return img.crop((0, 0, width, height - pixels))  

def convert_to_rgb(img):
    image_array = np.array(img)

    if len(image_array.shape) == 2:  # Check if grayscale
        image_array = np.stack([image_array] * 3, axis=2)


    return Image.fromarray(image_array)


def consistent_transform(frames, transform):
    # Set a fixed random seed for reproducibility
    seed = torch.randint(0, 10_000, (1,)).item()
    
    transformed_frames = []
    for frame in frames:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        transformed_frames.append(transform(frame))  # Apply same transformation
    
    return transformed_frames



####################################################
##### Transforms for Fundus Images #################
####################################################
TOL = 10
def crop_image_only_outside(img, tol=TOL):
    # img is 2D or 3D image data
    # tol  is tolerance
    mask = img > tol
    mask = mask[:-2,:-6]
    if img.ndim == 3:
        mask = mask.all(2)
    m, n = mask.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]
def pad_image_to_square(image, fill_color=(0, 0, 0)):
    """
    Pad an image to make it square with the minimal black border.
    Args:
    image (PIL.Image): The input image.
    fill_color (tuple): The color of the padding, default is black.
    Returns:
    PIL.Image: A new image that is padded to square dimensions.
    """
    # Crop the image first if necessary using your function
    image_array = crop_image_only_outside(np.array(image))  # Uncomment this if your function is available
    # Image.fromarray(image_array.astype('uint8')).show()
    # Determine the size of the padding
    height, width = image_array.shape[:2]
    if width == height:
        padded_array = image_array  # Already square
    elif width > height:
        padding = (width - height) // 2
        padded_array = np.full((width, width, image_array.shape[2]), fill_color, dtype=image_array.dtype)
        padded_array[padding:padding + height, :, :] = image_array
    else:
        padding = (height - width) // 2
        padded_array = np.full((height, height, image_array.shape[2]), fill_color, dtype=image_array.dtype)
        padded_array[:, padding:padding + width, :] = image_array
        # Convert the numpy array back to an image
    padded_image = Image.fromarray(padded_array.astype('uint8'))
    return padded_image
