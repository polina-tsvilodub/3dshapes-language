import os 
import json
import numpy as np
from random import shuffle
import torch
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer
from PIL import Image
from tqdm import tqdm
import random
from torchvision import transforms
import pickle
import h5py
    
class threeDshapes_Dataset(Dataset):
    """
    Dataset class for loading the dataset of images and captions from the 3dshapes dataset.

    Arguments:
    ---------
    path: str
        Path to directory containing all data.
    load_as_sandbox: bool
        Flag indicating whether to use the sandboxed dataset or apckage the full 3Dshapes dataset.
    num_labels: int
        Number of distinct captions to sample for each image. Relevant for using the dataloader for training models.
    labels_type: str
        "long" or "short". Indicates whether long or short captions should be used.
    run_inference: bool
        Flag indicating whether this dataset will be used for performing inference with a trained image captioner.
    batch_size: int
        Batch size. Has to be 1 in order to save the example image-caption pairs.
    vocab_file: str
        Name of vocab file.
    start_token: str
        Start token.
    end_token: str
        End token.
    unk_token: str
        Token to be used when encoding unknown tokens.
    pad_token: str
        Pad token to be used for padding captions tp max_sequence_length.
    max_sequence_length: int
        Length to which all captions are padded / truncated.
    """
    def __init__(
            self,
            path="data",  
            load_as_sandbox=True, # depending on this flag, check for existence of full dataset files
            num_labels=1, # number of ground truth labels to retrieve per image, influences resulting dataset size, check that it is <20/27
            labels_type="long", # alternative: short
            run_inference=False, # depending on this flag, check presence of model weights
            batch_size=1, 
            vocab_file="vocab.pkl", 
            start_token="START",  # might be unnecessary since vocab file is fixed anyways
            end_token="END",
            unk_token="UNK",
            pad_token="PAD", 
            max_sequence_length=25, # important for padding length
        ):
        # check vocab file exists
        assert os.path.exists(os.path.join(path, vocab_file)), "Make sure the vocab file exists in the directory passed to the dataloader (see README)"
        if load_as_sandbox:
            assert (os.path.exists(os.path.join(path, "sandbox_3Dshapes_1000.pkl")) and os.path.join(path, "sandbox_3Dshapes_resnet50_features_1000.pt")), "Make sure the sandbox dataset exists in the directory passed to the dataloader (see README)"
        # check presence of full files when load_assandbox = False
        if load_as_sandbox == False:
            assert os.path.exists(os.path.join(path, "3dshapes_all_ResNet_features.pt")), "Make sure to download the full ResNet image feature file (see README), check that the file name has NOT been modified and the file is located in the directory passed to the dataloader"
            assert os.path.exists(os.path.join(path, "3dshapes_captions_long.json")), "Make sure to download the full long captions file (see README), check that the file name has NOT been modified and the file is located in the directory passed to the dataloader"
            assert os.path.exists(os.path.join(path, "3dshapes_captions_short.json")), "Make sure to download the full short captions file (see README), check that the file name has NOT been modified and the file is located in the directory passed to the dataloader"
            assert (os.path.exists(os.path.join(path, "3dshapes.h5")) or ((os.path.exists(os.path.join(path, "3dshapes_images.npy")) and (os.path.exists(os.path.join(path, "3dshapes_labels.npy")))))), "Make sure to download the original 3Dshapes file or move the NumPy files to the directory passed to dataloader (see README)"

        if run_inference:
            assert os.path.exists(os.path.join(path, "pretrained_decoder_3dshapes.pkl")), "If you want to run inference with the pretrained image captioner, make sure to download the weights (see README), check that the file name has NOT been modified and the file is located in the directory passed to the dataloader"
        
        if labels_type == "long":
            assert num_labels <= 20, "Maximally 20 distinct image-long caption pairs can be created for one image"
        else:
            assert num_labels <= 27, "Maximally 27 distinct image-short caption pairs can be created for one image"
        # if user wishes to use the full dataset, create the np version of the images and labels if necessary 
        if load_as_sandbox == False:
            if not (os.path.exists(os.path.join(path, "3dshapes_images.npy")) and (os.path.exists(os.path.join(path, "3dshapes_labels.npy")))):
                print("WARNING: the data loader will create large NumPy files containing the 3Dshapes images and numeric labels!")
                # load original dataset
                raw_dataset = h5py.File(os.path.join(path, "3dshapes.h5"), "r")
                raw_images = raw_dataset['images']
                raw_labels = raw_dataset['labels']
                images_np = np.array(raw_images)
                labels_np = np.array(raw_labels)
                with open(os.path.join(path, "3dshapes_images.npy"), "wb") as f:
                    np.save(f, images_np)
                with open(os.path.join(path, "3dshapes_labels.npy"), "wb") as f:
                    np.save(f, labels_np)


        self.batch_size = batch_size
        with open(os.path.join(path, vocab_file), "rb") as vf:
            self.vocab = pickle.load(vf)

        self.max_sequence_length = max_sequence_length
        self.start_token = start_token
        self.end_token = end_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.tokenizer = get_tokenizer("basic_english")

        if load_as_sandbox:
            self.embedded_imgs = torch.load(os.path.join(path, "sandbox_3Dshapes_resnet50_features_1000.pt"))
            with open(os.path.join(path, "sandbox_3Dshapes_1000.pkl"), "rb") as f:
                self.sandbox_file = pickle.load(f)
                self.images = self.sandbox_file["images"]
                self.numeric_labels = self.sandbox_file["labels_numeric"]
                self.labels_long = self.sandbox_file["labels_long"]
                self.labels_short = self.sandbox_file["labels_short"]

        else:
            self.embedded_imgs = torch.load(os.path.join(path, "3dshapes_all_ResNet_features.pt")) 
            # load appropriate full datafiles
            self.images = np.load(
                os.path.join(path, "3dshapes_images.npy")
            )  
            self.numeric_labels = np.load(
                os.path.join(path, "3dshapes_labels.npy")
            )
            with open(os.path.join(path, "3d_shapes_captions_long.json"), "r") as fp:
                self.labels_long = json.load(fp)
            with open(os.path.join(path, "3d_shapes_captions_short.json"), "r") as fp:
                self.labels_short = json.load(fp)
            
           
        # TODO check that this is compatible with the full dataset
        # get the indices of the annotations (depending on the number of labels to use)
        if load_as_sandbox:
            if labels_type == "long":
                labels_ids_flat = [list(np.random.choice(range(len(self.labels_long[0])), num_labels, replace=False)) for i in range(len(self.images))]
                self.labels_flat = [self.labels_long[i][l] for i, sublst in enumerate(labels_ids_flat) for l in sublst]
                self.img_ids_flat = [id for id in range(len(self.images)) for i in range(num_labels)]
            else:
                labels_ids_flat = [list(np.random.choice(range(len(self.labels_short[0])), num_labels, replace=False)) for i in range(len(self.images))]
                self.labels_flat = [self.labels_short[i][l] for i, sublst in enumerate(labels_ids_flat) for l in sublst]
                self.img_ids_flat = [i for id in range(len(self.images)) for id in range(num_labels)]
        else:
            # TODO the exposed caption json files have to be recast to nested lists / dictionaries with numeric keys
            raise NotImplementedError

        print("len labels ids flat ", len(labels_ids_flat))
        print("len labels flat ", len(self.labels_flat), self.labels_flat[:5])
        print("len image ids flat ", len(self.img_ids_flat), self.img_ids_flat[:5])

    def __len__(self):
        """
        Returns length of dataset.
        """
        return len(self.img_ids_flat)

    def __getitem__(self, idx):
        """
        Iterator over the dataset.

        Arguments:
        ---------
        idx: int
            Index for accessing the flat image-caption pairs.
        
        Returns:
        -------
        target_img: np.ndarray (64,64,3)
            Original image.
        target_features: torch.Tensor(2048,)
            ResNet features of the image.
        target_lbl: str
            String caption.
        numeric_lbl: np.ndarray (6,)
            Original numeric image annotation.
        target_caption: torch.Tensor(batch_size, 25)
            Encoded caption.
        """
        # access raw image corresponding to the index in the entire dataset
        target_img = self.images[self.img_ids_flat[idx]]
        # access caption
        target_lbl = self.labels_flat[idx]
        # access original numeric annotation of the image
        numeric_lbl = self.numeric_labels[self.img_ids_flat[idx]]
        # cast type
        target_img = np.asarray(target_img).astype('uint8')
        # retrieve ResNet features, accessed through original image ID
        target_features = self.embedded_imgs[self.img_ids_flat[idx]]
        # tokenize label
        tokens = self.tokenizer(str(target_lbl).lower().replace("-", " "))
        # Convert caption to tensor of word ids, append start and end tokens.
        target_caption = self.tokenize_caption(tokens)
        # convert to tensor
        target_caption = torch.Tensor(target_caption).long()
        
        return target_img, target_features, target_lbl, numeric_lbl, target_caption

    def tokenize_caption(self, label):
        """
        Helper for converting list of tokens into list of token IDs.
        Expects tokenized caption as input.

        Arguments:
        --------
        label: list
            Tokenized caption.
        
        Returns:
        -------
        tokens: list
            List of token IDs, prepended with start, end, padded to max length.
        """
        label = label[:self.max_sequence_length]
        tokens = [self.vocab["word2idx"][self.start_token]]
        for t in label:
            try:
                tokens.append(self.vocab["word2idx"][t])
            except:
                tokens.append(self.vocab["word2idx"][self.unk_token])
        tokens.append(self.vocab["word2idx"][self.end_token])
        # pad
        while len(tokens) < self.max_sequence_length:
            tokens.append(self.vocab["word2idx"][self.pad_token])

        return tokens

    def get_labels_for_image(self, id, caption_type="long"):
        """
        Helper for getting all annotations for a given image id.

        Arguments:
        ---------
        id: int
            Index of image caption pair containing the image
            for which the full list of captions should be returned.
        caption_type: str
            "long" or "short". Indicates type of captions to provide.

        Returns:
        -------
            List of all captions for given image.
        """
        if caption_type == "long":
            return self.labels_long[self.img_ids_flat[id]]
        else:
            return self.labels_short[self.img_ids_flat[id]]

def get_loader(
    path="data", 
    load_as_sandbox=True, # depending on this flag, check for existence of full dataset files
    num_labels=1, # number of ground truth labels to retrieve per image, influences resulting dataset size, check that it is <20/27
    labels_type="long", # alternative: short
    run_inference=False, # depending on this flag, check presence of model weights
    batch_size=1, 
    vocab_file="vocab.pkl", 
    start_token="START",  # might be unnecessary since vocab file is fixed anyways
    end_token="END",
    unk_token="UNK",
    pad_token="PAD", 
    max_sequence_length=25,
):
    """
    Returns the data loader.
    Arguments:
    ----------
    path: str
        Path to directory containing all data.
    load_as_sandbox: bool
        Flag indicating whether to use the sandboxed dataset or apckage the full 3Dshapes dataset.
    num_labels: int
        Number of distinct captions to sample for each image. Relevant for using the dataloader for training models.
    labels_type: str
        "long" or "short". Indicates whether long or short captions should be used.
    run_inference: bool
        Flag indicating whether this dataset will be used for performing inference with a trained image captioner.
    batch_size: int
        Batch size. Has to be 1 in order to save the example image-caption pairs.
    vocab_file: str
        Name of vocab file.
    start_token: str
        Start token.
    end_token: str
        End token.
    unk_token: str
        Token to be used when encoding unknown tokens.
    pad_token: str
        Pad token to be used for padding captions tp max_sequence_length.
    max_sequence_length: int
        Length to which all captions are padded / truncated.
        
    Returns:
    --------
      data_loader: Torch DataLoader.
    """

    dataset = threeDshapes_Dataset(
        path=path, 
        load_as_sandbox=load_as_sandbox, # depending on this flag, check for existence of full dataset files
        num_labels=num_labels, # number of ground truth labels to retrieve per image, influences resulting dataset size, check that it is <20/27
        labels_type=labels_type, # alternative: short
        run_inference=run_inference, # depending on this flag, check presence of model weights
        
        batch_size=batch_size, 
        vocab_file=vocab_file, 
        start_token=start_token,  # might be unnecessary since vocab file is fixed anyways
        end_token=end_token,
        unk_token=unk_token,
        pad_token=pad_token, 
        max_sequence_length=max_sequence_length,
    )

    # Create and assign a batch sampler to retrieve a batch with the sampled indices.
    initial_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=list(range(batch_size)))
    # data loader 
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_sampler=torch.utils.data.sampler.BatchSampler(sampler=initial_sampler,
                                                            batch_size=dataset.batch_size,
                                                            drop_last=False)
    )

    return data_loader