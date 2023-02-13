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
        Flag indicating whether to use the sandboxed dataset or package the full 3Dshapes dataset.
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
    pairs: str
        Indicator for whether the dataloader is returning single images (when "none"), random image pairs ("random") or similar image pairs ("similar").
    id_file: str
        Path to file containing IDs from which dataloader should be constructed.
    features: str
        String of features which should be held fixed among similar image pairs. If not set, random subsets will be sampled.
    full_ds_path: str
        Path to directory containing data for loading the full dataset.
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
            start_token="START",  
            end_token="END",
            unk_token="UNK",
            pad_token="PAD", 
            max_sequence_length=25, # important for padding length
            pairs="none", 
            id_file=None,
            features=None,
            full_ds_path="data",
        ):

        # check vocab file exists
        assert os.path.exists(os.path.join(path, vocab_file)), "Make sure the vocab file exists in the directory passed to the dataloader (see README)"
        if load_as_sandbox:
            assert (os.path.exists(os.path.join(path, "sandbox_3Dshapes_1000.pkl")) and os.path.join(path, "sandbox_3Dshapes_resnet50_features_1000.pt")), "Make sure the sandbox dataset exists in the directory passed to the dataloader (see README)"
        # check presence of full files when load_assandbox = False
        if load_as_sandbox == False:
            assert os.path.exists(os.path.join(full_ds_path, "3dshapes_all_ResNet_features.pt")), "Make sure to download the full ResNet image feature file (see README), check that the file name has NOT been modified and the file is located in the 'data' directory on the same level as the sanbox directory."
            assert os.path.exists(os.path.join(full_ds_path, "3dshapes_captions_long.json")), "Make sure to download the full long captions file (see README), check that the file name has NOT been modified and the file is located in the 'data' directory on the same level as the sanbox directory."
            assert os.path.exists(os.path.join(full_ds_path, "3dshapes_captions_short.json")), "Make sure to download the full short captions file (see README), check that the file name has NOT been modifiedand the file is located in the 'data' directory on the same level as the sanbox directory."
            assert (os.path.exists(os.path.join(full_ds_path, "3dshapes.h5")) or ((os.path.exists(os.path.join(full_ds_path, "3dshapes_images.npy")) and (os.path.exists(os.path.join(full_ds_path, "3dshapes_labels.npy")))))), "Make sure to download the original 3Dshapes file or move the NumPy files to the 'data' directory on the same level as the sanbox directory (see README)"

        if run_inference:
            assert os.path.exists(os.path.join(path, "pretrained_decoder_3dshapes.pkl")), "If you want to run inference with the pretrained image captioner, make sure to download the weights (see README), check that the file name has NOT been modified and the file is located in the directory passed to the dataloader"
        
        if labels_type == "long":
            assert num_labels <= 20, "Maximally 20 distinct image-long caption pairs can be created for one image"
        else:
            assert num_labels <= 27, "Maximally 27 distinct image-short caption pairs can be created for one image"
        # if user wishes to use the full dataset, create the np version of the images and labels if necessary 
        if load_as_sandbox == False:
            if not (os.path.exists(os.path.join(full_ds_path, "3dshapes_images.npy")) and (os.path.exists(os.path.join(full_ds_path, "3dshapes_labels.npy")))):
                print("WARNING: the data loader will create large NumPy files containing the 3Dshapes images and numeric labels!")
                # load original dataset
                raw_dataset = h5py.File(os.path.join(full_ds_path, "3dshapes.h5"), "r")
                raw_images = raw_dataset['images']
                raw_labels = raw_dataset['labels']
                images_np = np.array(raw_images)
                labels_np = np.array(raw_labels)
                with open(os.path.join(full_ds_path, "3dshapes_images.npy"), "wb") as f:
                    np.save(f, images_np)
                with open(os.path.join(full_ds_path, "3dshapes_labels.npy"), "wb") as f:
                    np.save(f, labels_np)

        self.batch_size = batch_size
        with open(os.path.join(path, vocab_file), "rb") as vf:
            self.vocab = pickle.load(vf)

        ##### additions for evaluating the retrained agents ####
        self.categories = {
            'floor_hue': [0,0.1,0.2,0.30000000000000004,0.4,0.5,0.6000000000000001,0.7000000000000001,0.8,0.9],
            'wall_hue': [0,0.1,0.2,0.30000000000000004,0.4,0.5,0.6000000000000001,0.7000000000000001,0.8,0.9], 
            'object_hue': [0,0.1,0.2,0.30000000000000004,0.4,0.5,0.6000000000000001,0.7000000000000001,0.8,0.9],
            'scale': [0.75,0.8214285714285714,0.8928571428571428,0.9642857142857143,1.0357142857142856,1.1071428571428572,1.1785714285714286,1.25], 
            'shape': [0,1,2,3], 
            'orientation': [-30,-25.714285714285715,-21.42857142857143,-17.142857142857142,-12.857142857142858,-8.571428571428573,-4.285714285714285,0, 4.285714285714285,8.57142857142857,12.857142857142854,17.14285714285714,21.42857142857143,25.714285714285715,30],
        }
        with open(os.path.join(path, "categories2imgIDs_3dshapes_fixed_float_val.json"), "r") as fp:
            self.cats2imgIDs = json.load(fp)

        #######

        self.max_sequence_length = max_sequence_length
        self.start_token = start_token
        self.end_token = end_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.tokenizer = get_tokenizer("basic_english")
        self.pairs = pairs
        self.features = sorted(features.split(",")) if features else []

        if load_as_sandbox:
            self.embedded_imgs = torch.load(os.path.join(path, "sandbox_3Dshapes_resnet50_features_1000.pt"))
            with open(os.path.join(path, "sandbox_3Dshapes_1000.pkl"), "rb") as f:
                self.sandbox_file = pickle.load(f)
                self.images = self.sandbox_file["images"]
                self.numeric_labels = self.sandbox_file["labels_numeric"]
                self.labels_long = self.sandbox_file["labels_long"]
                self.labels_short = self.sandbox_file["labels_short"]

        else:
            self.embedded_imgs = torch.load(os.path.join(full_ds_path, "3dshapes_all_ResNet_features.pt")) 
            # load appropriate full datafiles
            self.images = np.load(
                os.path.join(full_ds_path, "3dshapes_images.npy")
            )  
            self.numeric_labels = np.load(
                os.path.join(full_ds_path, "3dshapes_labels.npy")
            )
            with open(os.path.join(full_ds_path, "3dshapes_captions_long.json"), "r") as fp:
                self.labels_long = json.load(fp, object_hook=parse_int_keys)
            with open(os.path.join(full_ds_path, "3dshapes_captions_short.json"), "r") as fp:
                self.labels_short = json.load(fp, object_hook=parse_int_keys)
        
        if id_file:
            self.id_file = torch.load(id_file)
            shuffle(self.id_file)
        else:
            self.id_file = list(range(len(self.images)))

        # get the indices of the annotations (depending on the number of labels to use)
        if self.id_file:
            if labels_type == "long":
                labels_ids_flat = [list(np.random.choice(range(len(self.labels_long[0])), num_labels, replace=False)) for i in self.id_file]
                self.labels_flat = [self.labels_long[i][l] for i, sublst in list(zip(self.id_file, labels_ids_flat)) for l in sublst]
                self.img_ids_flat = [id for id in self.id_file for i in range(num_labels)]
            else:
                labels_ids_flat = [list(np.random.choice(range(len(self.labels_short[0])), num_labels, replace=False)) for i in self.id_file]
                self.labels_flat = [self.labels_short[i][l] for i, sublst in list(zip(self.id_file, labels_ids_flat)) for l in sublst]
                self.img_ids_flat = [id for id in self.id_file for i in range(num_labels)]

        else:
            if labels_type == "long":
                labels_ids_flat = [list(np.random.choice(range(len(self.labels_long[0])), num_labels, replace=False)) for i in range(len(self.images))]
                self.labels_flat = [self.labels_long[i][l] for i, sublst in enumerate(labels_ids_flat) for l in sublst]
                self.img_ids_flat = [id for id in range(len(self.images)) for i in range(num_labels)]
            else:
                labels_ids_flat = [list(np.random.choice(range(len(self.labels_short[0])), num_labels, replace=False)) for i in range(len(self.images))]
                self.labels_flat = [self.labels_short[i][l] for i, sublst in enumerate(labels_ids_flat) for l in sublst]
                self.img_ids_flat = [id for id in range(len(self.images)) for i in range(num_labels)]
       

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
        if self.pairs == "similar":
            # access raw image corresponding to the index in the entire dataset
            target_img = self.images[self.img_ids_flat[idx[0]]]
            dist_img = self.images[self.img_ids_flat[idx[1]]]
            target_features = self.embedded_imgs[self.img_ids_flat[idx[0]]]
            dist_features = self.embedded_imgs[self.img_ids_flat[idx[1]]]
            # access caption
            target_lbl = self.labels_flat[idx[0]]
            # access caption
            dist_lbl = self.labels_flat[idx[1]]
            # tokenize label
            tokens = self.tokenizer(str(target_lbl).lower().replace("-", " "))
            # Convert caption to tensor of word ids, append start and end tokens.
            target_caption = self.tokenize_caption(tokens)
            # convert to tensor
            target_caption = torch.Tensor(target_caption).long()
            # tokenize label
            tokens_dist = self.tokenizer(str(dist_lbl).lower().replace("-", " "))
            # Convert caption to tensor of word ids, append start and end tokens.
            dist_caption = self.tokenize_caption(tokens_dist)
            # convert to tensor
            dist_caption = torch.Tensor(dist_caption).long()
            return target_img, dist_img, target_features, dist_features, target_lbl, dist_lbl, target_caption, dist_caption
        
        else:
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
        label = label[:(self.max_sequence_length-2)]
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

    def get_func_similar_train_indices(self, i_step):
            """
            Naive function constructing pairs of similar images for reference game style training / inference. 
            Returns a list of tuples consisting of a target and distractor index. 
            Allows to select features on which the pairs should match; if not provided three features
            are sampled per batch.

            Returns:
            -------
                list: (batch_size, int, int)
                    List of tuples of target and distractor indices, each for a single reference game iteration.
            """

            # get ann / img ID slice
            target_img_ids = self.img_ids_flat[(i_step-1)*self.batch_size : i_step*self.batch_size]
            target_labels = [self.numeric_labels[int(i)] for i in target_img_ids]
            # select at random 3 categories along which the image should be constant in this batch
            if not self.features:
                sel_categories = np.random.choice(list(self.categories.keys()), size=3, replace=False) 
            else:
                sel_categories = self.features
            sel_categories_inds = [list(self.categories.keys()).index(c) for c in sel_categories]
            # create a container for the distractor indices
            distractor_inds = []
            
            # iterate over each sample to find the right distractor
            for lbl in target_labels:
                # create a placeholder for intersecting the indices for that 
                possible_inds = []
                # get values of sampled fixed categories and matching distractor indices
                for i, c_i in enumerate(list(zip(sel_categories, sel_categories_inds))):
                    if i == 0:
                        possible_inds = self.cats2imgIDs[c_i[0]][str(lbl[c_i[1]])]
                        
                    else:
                        possible_inds = list(set.intersection(set(self.cats2imgIDs[c_i[0]][str(lbl[c_i[1]])]), set(possible_inds)))
                        
                # get one distractor 
                j = 0
                distractor_ann_ind = []
                try:
                    while len(distractor_ann_ind) < 1 and j < len(possible_inds)-1:
                        
                        j += 1
                        distractor_img_ind = possible_inds[j]
                        # convert retrieved distractor image index into distractor annotation index
                        distractor_ann_ind = np.where([self.img_ids_flat[i] == distractor_img_ind for i in range(len(self.id_file))])[0]
                        
                        try:
                            distractor_inds.append(int(distractor_ann_ind[0]))
                        except IndexError:
                            continue        
                except IndexError:
                    raise Exception                            
            
            inds_tuples = list(zip(list(range((i_step-1)*self.batch_size, i_step*self.batch_size)), distractor_inds)) 
            
            return inds_tuples


def parse_int_keys(dct):
    """
    Helper function for converting full caption json files into 
    numeric keys for parsing with the same function as the sandbox.
    """
    rval = dict()
    for key, val in dct.items():
        try:
            # Convert the key to an integer
            int_key = int(key)
            # Assign value to the integer key in the new dict
            rval[int_key] = val
        except ValueError:
            # Couldn't convert key to an integer; Use original key
            rval[key] = val
    return rval

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
    pairs="none", 
    id_file=None,
    features=None,
    full_ds_path="data",
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
    pairs: str
        Indicator for whether the dataloader is returning single images (when "none"), random image pairs ("random") or similar image pairs ("similar").
    id_file: str
        Path to file containing IDs from which dataloader should be constructed.
    features: str
        String of features which should be held fixed among similar image pairs. If not set, random subsets will be sampled.
    full_ds_path: str
        Path to directory containing data for loading the full dataset.

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
        pairs=pairs, 
        id_file=id_file,
        features=features,
        full_ds_path=full_ds_path,
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