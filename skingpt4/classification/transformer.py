import torchvision.transforms as transforms

class Transformer:
    def __init__(self):
        self.transforms = [ transforms.Resize((1080, 810))]

    def downsample(self, downsample_factor: float) -> None:
        """
        Add transformations that downsample the image by a certain factor and enforce 1080x810 size

        Args:
            downsample_factor (float): The factor by which to downsample the image

        """
        h = downsample_factor * 1080
        w = downsample_factor * 810
        downsample_transforms = [
            transforms.Resize((h, w)),
            transforms.Resize((1080, 810))
        ]

        self.transforms = downsample_transforms + self.transforms

    def randomize_img(self, degree = 1) -> None:
        """
        Add transformations that introduce image variation (flips, color jitter)

        Args:
            degree (int): The degree of variation to introduce (affects magnitude of color jittering only)
        """
        random_transforms = [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(brightness=0.1*degree, contrast=0.1*degree, saturation=0.1*degree, hue=0.25*degree)
        ]

        self.transforms = self.transforms + random_transforms

    def to_tensor(self) -> None:
        """
        Add transformation that converts the image to a tensor
        """
        self.transforms = self.transforms + [transforms.ToTensor()]

    def get_transforms(self) -> transforms.Compose:
        """
        Return all transformations from this class instance

        Returns:
            transforms.Compose: The composed transformations
        """
        return transforms.Compose(self.transforms)