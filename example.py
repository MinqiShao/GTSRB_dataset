from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize
import cv2


transform_train = Compose([
        ToPILImage(),
        Resize((32, 32)),
        ToTensor()
    ])
trainset = DatasetFolder(
        root=osp.join(datasets_root_dir, 'GTSRB_FINAL', 'train'),  # please replace this with path to your training set
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)


transform_test = Compose([
        ToPILImage(),
        Resize((32, 32)),
        ToTensor()
    ])
 testset = DatasetFolder(
        root=osp.join(datasets_root_dir, 'GTSRB_FINAL', 'test'),  # please replace this with path to your test set
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)

train_loader = DataLoader(
        trainset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        drop_last=False,
        pin_memory=True
    )

