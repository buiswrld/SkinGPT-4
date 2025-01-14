import torchvision.transforms as transforms

class Transformer:
    def __init__(self):
        self.transforms = [ transforms.Resize((810, 1080)),transforms.ToTensor()]

    def downsample(self, downsample_dim, starting_img_size):
        downsample_dim = downsample_dim
        starting_img_size = starting_img_size
        downsample_transforms = [
            transforms.Resize((downsample_dim, downsample_dim)),
            transforms.Resize((starting_img_size, starting_img_size))
        ]

        self.transforms = downsample_transforms + self.transforms

    def randomize(self):
        random_transforms = [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        ]

        self.transforms = self.transforms + random_transforms