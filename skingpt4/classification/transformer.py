import torchvision.transforms as transforms

class Transformer:
    def __init__(self):
        self.transforms = [ transforms.Resize((1080, 810))]

    def downsample(self, downsample_factor: float) -> None:
        h = downsample_factor * 1080
        w = downsample_factor * 810
        downsample_transforms = [
            transforms.Resize((h, w)),
            transforms.Resize((1080, 810))
        ]

        self.transforms = downsample_transforms + self.transforms

    def randomize_img(self, degree = 1) -> None:
        random_transforms = [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(brightness=0.1*degree, contrast=0.1*degree, saturation=0.1*degree, hue=0.25*degree)
        ]

        self.transforms = self.transforms + random_transforms

    def to_tensor(self) -> None:
        self.transforms = self.transforms + [transforms.ToTensor()]

    def transforms(self) -> transforms.Compose:
        return transforms.Compose(self.transforms)