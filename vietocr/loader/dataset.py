from torch.utils.data import Dataset
from typing import Optional
import torchvision.transforms as transforms
import os
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
from VietOCR.vietocr.loader.aug import ImgAugTransform
from VietOCR.vietocr.model.vocab import Vocab
from torch.utils.data.sampler import Sampler
import random
import torch

########################################################## translate.py ##########################################################
import math
import numpy as np

image_height = 32
image_min_width = 32
image_max_width = 512

def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w/round_to)*round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height

def process_image(image, image_height=image_height, image_min_width=image_min_width, image_max_width=image_max_width):
    img = image.convert('RGB')

    w, h = img.size
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)

    img = img.resize((new_w, image_height), Image.LANCZOS)

    img = np.asarray(img)#.transpose(2, 0, 1)
    # img = img/255 # not necessary because transforms.ToTensor() automatically do it
    return img

########################################################## translate.py ##########################################################

class OCRDataset(Dataset):
    def __init__(self, 
        data_dir: str, 
        train_gt_path: str,
        image_height: int = 32, 
        image_min_width: int = 32, 
        image_max_width: int = 512
    ) -> None:
        self.data_dir: str = data_dir
        self.samples: list[dict] = self.load_data(train_gt_path)

        self.image_height = image_height
        self.image_min_width = image_min_width
        self.image_max_width = image_max_width

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index) -> tuple:
        # get sample at index
        sample: dict = self.samples[index]

        # get sample's information
        filename = sample["filename"]
        word = sample["word"]

        # open & process image
        image_path = os.path.join(self.data_dir, filename)
        image = Image.open(image_path).convert("RGB")
        image = process_image(image, self.image_height, self.image_min_width, self.image_max_width)

        # return image, word
        return {'filename': filename, 'image': image, 'word': word}

    def load_data(self, train_gt_path) :
        # init list
        samples = []

        # read file train_gt.txt
        with open(train_gt_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    filename, word = parts
                    sample_dict = {'filename': filename, 'word': word}
                    samples.append(sample_dict)

        return samples

class OCRGenDataset(Dataset):
    def __init__(
        self,
        dataset: OCRDataset,
        augment = ImgAugTransform
    ):
        self.dataset = dataset
        self.augment = augment
    
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> tuple:
        # get sample from dataset
        sample = self.dataset[index]
        filename = sample['filename']
        image = sample['image']
        word = sample['word']

        # augment
        image = self.augment(image)
        image = np.asarray(image)

        # return image, word
        return {'filename': filename, 'image': image, 'word': word}

class OCRTransformedDataset(Dataset):
    def __init__(
        self, 
        dataset: OCRDataset,
        vocab: Vocab,
        transform: Optional[transforms.Compose] = None
    ) -> None:
        self.dataset = dataset
        self.vocab = vocab
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform
        self.build_cluster_indices() ##add here

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> tuple:
        # get sample from dataset
        sample = self.dataset[index]
        filename = sample['filename']
        image = sample['image']
        word = sample['word']

        # transform for input
        # print("____________________", type(image))
        transformed_image = self.transform(image.copy())

        # transform for output
        label = self.vocab.encode(word)

        return transformed_image, label, filename
    
    def build_cluster_indices(self):
        self.cluster_indices = defaultdict(list)

        for i in tqdm(range(self.__len__()), desc = "Building cluster indices"):
            bucket = self.get_bucket(i)
            self.cluster_indices[bucket].append(i)
    
    def get_bucket(self, idx):
        # get sample at index
        sample = self.dataset[idx]
        image = sample['image']
        word = sample['word']

        height, width, channel = image.shape

        return width
    
class ClusterRandomSampler(Sampler):
    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle        

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):
        batch_lists = []
        # self.data_source.build_cluster_indices() ## dict[width] = [idx1, idx2, ...] ##comment here
        for cluster, cluster_indices in self.data_source.cluster_indices.items():
            if self.shuffle:
                random.shuffle(cluster_indices)

            batches = [cluster_indices[i:i + self.batch_size] for i in range(0, len(cluster_indices), self.batch_size)]
            batches = [_ for _ in batches if len(_) == self.batch_size]
            if self.shuffle:
                random.shuffle(batches)

            batch_lists.append(batches)

        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)

        lst = self.flatten_list(lst)

        return iter(lst)

    def __len__(self):
        return len(self.data_source)

class Collator(object):
    def __init__(self, masked_language_model=False):
        self.masked_language_model = masked_language_model

    def __call__(self, batch):
        filenames = []
        img = []
        target_weights = []
        tgt_input = [] ##target input
        max_label_len = max(len(sample[1]) for sample in batch)
        for sample in batch:
            filenames.append(sample[2])
            img.append(sample[0])
            label = sample[1]
            label_len = len(label)
            
            
            tgt = np.concatenate((
                label,
                np.zeros(max_label_len - label_len, dtype=np.int32)))
            tgt_input.append(tgt)

            one_mask_len = label_len - 1

            target_weights.append(np.concatenate((
                np.ones(one_mask_len, dtype=np.float32),
                np.zeros(max_label_len - one_mask_len,dtype=np.float32))))


        tgt_input = np.array(tgt_input, dtype=np.int64).T
        tgt_output = np.roll(tgt_input, -1, 0).T
        tgt_output[:, -1]=0
        
        # random mask token
        if self.masked_language_model:
            mask = np.random.random(size=tgt_input.shape) < 0.05
            mask = mask & (tgt_input != 0) & (tgt_input != 1) & (tgt_input != 2)
            tgt_input[mask] = 3

        tgt_padding_mask = np.array(target_weights)==0

        rs = {
            'img': torch.stack(img, dim=0),
            'filenames': filenames,
            'tgt_input': torch.LongTensor(tgt_input),
            'tgt_output': torch.LongTensor(tgt_output),
            'tgt_padding_mask': torch.BoolTensor(tgt_padding_mask)
        }   
        
        return rs
if __name__ == "__main__":
    import torch

    # test dataset
    print("\nTEST DATASET\n")
    data_dir = "/Users/tiendzung/Project/vietocr-git/thao_train"
    train_gt_path = "/Users/tiendzung/Project/vietocr-git/thao_train/train_gt.txt"
    vocab = Vocab()

    dataset = OCRDataset(
        data_dir=data_dir,
        train_gt_path=train_gt_path,
    )
    image, word = dataset[0]
    # print(len(dataset))
    assert len(dataset) == 103000, "Dataset should have 103000 samples"
    assert isinstance(image, np.ndarray), "Image should be a numpy array"
    assert isinstance(word, str), "Word should be a string"

    print(f"Word : {word}")
    print(f"Image shape: {image.shape}")
    print(f"Image : {image}")

    # test transformed dataset
    print("\nTEST DATAMODULE\n")
    trans_dataset = OCRTransformedDataset(
        dataset=dataset,
        vocab=Vocab()
    )

    image, label = trans_dataset[0]

    assert len(trans_dataset) == 103000, "Transformed dataset should have 103000 samples"
    assert isinstance(image, torch.Tensor), "Image should be a torch Tensor"
    assert isinstance(label, list), "Label should be a list of int"

    print(f"Label : {label}")
    print(f"Image shape: {image.shape}")
    print(f"Image : {image}")