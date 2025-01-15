import torchvision.transforms as transforms

class Transformer:
    def __init__(self):
        self.transforms = [ transforms.Resize((810, 1080))]

    # TODO ~ Support scaling by 810*1080, i.e. nonsquare downsample_dim
    def downsample(self, downsample_dim):
        downsample_dim = downsample_dim
        downsample_transforms = [
            transforms.Resize((downsample_dim, downsample_dim)),
            transforms.Resize((810, 1080))
        ]

        self.transforms = downsample_transforms + self.transforms

    def randomize_img(self, degree = 1):
        random_transforms = [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(brightness=0.1*degree, contrast=0.1*degree, saturation=0.1*degree, hue=0.25*degree)
        ]

        self.transforms = self.transforms + random_transforms

    def to_tensor(self):
        self.transforms = self.transforms + [transforms.ToTensor()]

    def transforms(self):
        return transforms.Compose(self.transforms)