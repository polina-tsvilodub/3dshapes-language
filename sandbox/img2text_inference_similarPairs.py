"""
Script for running inference on trained spekaer model, for testing purposes
on the image captioners from the thesis.
"""
import torch
from utils.dataset_utils_similarPairs import get_loader
from utils.DecoderRNN import DecoderRNN
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
import pandas as pd
import random
from tqdm import tqdm 

def main(
    path="data", 
    run_inference=False, 
    load_as_sandbox=True,
    num_labels=1,
    labels_type="long",
    batch_size=1,
    max_sequence_length=25,
    vocab_file="vocab.pkl",
    pairs="none",
    id_file=None,
    features=None,
    pairs_file=None,
    full_ds_path="data",
    output_path="3dshapes_example.png",
):
    """
    Entry point for displaying example 3Dshapes images and ground truth captions,
    possibly with image captioner predictions.

    Arguments:
    ---------
    path: str
        Name of directory where required files (from repository and/or for full inference) are located.
    run_inference: bool
        Flag indicating whether to run inference on the image captioner.
    load_as_sandbox: bool
        Flag indicating whether to load the snadboxed dataset from the repository, or download and package the full 3Dshapes dataset.
    num_labels: int
        Number of distinct captions to sample for each image. Relevant for using the dataloader for training models.
    labels_type: str
        "long" or "short". Indicates whether long or short captions should be used.
    batch_size: int
        Batch size. Has to be 1 in order to save the example image-caption pairs.
    max_sequence_length: int
        Length to which all captions are padded / truncated for training (full captions are displayed in the example png regardless of the setting).
    vocab_file: str
        Name of vocab file.
    pairs: str
        Indicator for whether the dataloader is returning single images (when "none"), random image pairs ("random") or similar image pairs ("similar").
    id_file: str
        Path to file containing IDs from which dataloader should be constructed.
    features: str
        String of features which should be held fixed among similar image pairs. If not set, random subsets will be sampled.
    full_ds_path: str
        Path to directory containing data for loading the full dataset.
    """
    torch.manual_seed(1234)
    random.seed(1234)

    if features:
        assert pairs == "similar", "You can only match images by set features when retrieving pairs=similar images."
        assert all([f in ["shape", "object_hue", "scale", "floor_hue", "orientation", "wall_hue"] for f in features.split(",")]), "The features have to be a subset of  ['shape', 'object_hue', 'scale', 'floor_hue', 'orientation', 'wall_hue']"

    if pairs == "similar":
        assert load_as_sandbox == False, "Similar image pairs can only be retrieved from full dataset or provided IDs file when load_as_sandbox=False."
    if pairs_file:
        assert load_as_sandbox == False, "File containing image pair indices can only be used when load_as_sandbox=False"
    #### define inference parameters ####
    # number of steps to run inference / retrieve examples for
    num_batches = 10
    num_batches_buffer = 30 # buffer in case there is no matching similar distractor for a given features set / image ID subsample 
    
    # name of trained models
    decoder_file = "pretrained_decoder_3dshapes.pkl"

    # decoder configs
    embed_size = 512
    visual_embed_size = 512
    hidden_size = 512
    # define data loader
    data_loader_test = get_loader(
        load_as_sandbox=load_as_sandbox,
        run_inference=run_inference,
        num_labels=num_labels,
        labels_type=labels_type,
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
        vocab_file=vocab_file,
        pairs=pairs, 
        id_file=id_file,
        features=features,
        full_ds_path=full_ds_path,
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # test if MPS device can be used with loaded package versions
    if device == "mps":
        tensor1 = torch.randn(3, 2)
        tensor2 = torch.randn(3, 2)
        tensor1 = tensor1.to(device)
        tensor2 = tensor2.to(device)
        try:
            test_res = torch.mm(tensor1, tensor2)
        except:
            # if errors occur, switch back to cpu
            print("Your current library and / or OS versions don't support inference on MPS, switching back to CPU")
            device = "cpu"

    vocab_size = len(data_loader_test.dataset.vocab["word2idx"].keys())
    
    if run_inference:
        decoder = DecoderRNN(embed_size, hidden_size, vocab_size, visual_embed_size)
        # if in train mode, pure sampling runs, in eval mode greedy sampling runs
        decoder.eval()

        # Load the trained weights.
        decoder.load_state_dict(torch.load(os.path.join(path, decoder_file), map_location=torch.device('cpu'))) 
    
        # Move models to GPU if available.
        decoder.to(device)

    if pairs != "none":
        fig, axs = plt.subplots(nrows=num_batches, ncols=2, figsize=(25, 35))
    else:
        fig, axs = plt.subplots(nrows=num_batches, ncols=1, figsize=(15, 35))

    def clean_sentence(output):
        """
        Helper function for visualization purposes.
        Transforms list of token indices to a sentence. 
        Also accepts mulit-dim tensors (for batch size > 1).

        Args:
        ----
        output: torch.Tensor(batch_size, sentence_length)
            Tensor representing sentences in form of token indices.

        Returns:
        -------
        sentence: str
            String representing decoded sentences in natural language.
        """
        list_string = []
        
        for idx in output:
            for i in idx:
                try:
                    list_string.append(data_loader_test.dataset.vocab["idx2word"][i.item()])
                except ValueError:
                    for y in i:
                        list_string.append(data_loader_test.dataset.vocab["idx2word"][y.item()])
        sentence = ' '.join(list_string) # Convert list of strings to full string
        sentence = sentence.capitalize()  # Capitalize the first letter of the first word
        return sentence

    # if a file containing a list of target-distractor index pairs was passed, use those pairs
    if pairs_file:
        pairs_list = torch.load(pairs_file)
    else:
        pairs_list = np.random.choice(len(data_loader_test.dataset), num_batches_buffer)
    

    exception_counter = 0
    completed_counter = 0
    # for desired number of batches (should match the size of plot), run inference, build plot
    for i, id in tqdm(enumerate(pairs_list)):
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        if not pairs_file:
            if pairs == "similar":
                similar_ids = data_loader_test.dataset.get_func_similar_train_indices(id)
            else:
                similar_ids = np.random.choice(len(pairs_list), batch_size)
        else:
            similar_ids = [id]
        
        new_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=similar_ids) 
        data_loader_test.batch_sampler.sampler = new_sampler        

        # get the batch
        try:
            if pairs == "similar":
                target_img, dist_img, target_features, dist_features, target_lbl, dist_lbl, target_caption, dist_caption = next(iter(data_loader_test)) 
                completed_counter += 1
                dist_caption = dist_caption.to(device)
                dist_features = dist_features.to(device)
               
            else:
                target_img, target_features, target_lbl, numeric_lbl, target_caption = next(iter(data_loader_test))
                completed_counter += 1
        except:
            exception_counter += 1
            continue

        ##### get random distractor
        if pairs == "random":
            new_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=np.random.choice(len(data_loader_test.dataset), 1).tolist())
            data_loader_test.batch_sampler.sampler = new_sampler
            dist_img, dist_features, dist_lbl, _, dist_caption = next(iter(data_loader_test)) 
            dist_caption = dist_caption.to(device)
            dist_features = dist_features.to(device)

        # print("All short captions for images ID ", id, data_loader_test.dataset.get_labels_for_image(id, caption_type='short'))
        
        if run_inference:
            # move everything to device
            target_caption = target_caption.to(device)
            target_features = target_features.to(device)
            
            
            # duplicate target since pretrained model expects two images as input
            if pairs != "none":
                both_images = torch.cat((target_features.unsqueeze(1), dist_features.unsqueeze(1)), dim=1)
            else:
                both_images = torch.cat((target_features.unsqueeze(1), target_features.unsqueeze(1)), dim=1)


            # run prediction (auto-regressive decoding)
            output, _, _, _ = decoder.sample(both_images, target_caption.shape[-1]-1)
            # transform idx prediction to string
            sentence_t = clean_sentence(output)
            
            # find index of end token for displaying 
            if "end" in sentence_t:
                len_sentence_t = sentence_t.split(" ").index("end")
            else:
                len_sentence_t = len(sentence_t.split(" "))
            
            cleaned_sentence_t = " ".join(sentence_t.split()[:len_sentence_t]).replace(" pad", "")
            
        
        # show the images, ground truth captions and predicted captions
        if batch_size == 1:
            if run_inference:
                title = "Ground truth:\n " + target_lbl[0] + "\nPredicted caption:\n " + cleaned_sentence_t
            else:
                title = "Ground truth:\n " + target_lbl[0] 

            if pairs == "none":
                axs[completed_counter-1].imshow(target_img.squeeze(0).permute(0,1,2))
                axs[completed_counter-1].set_title(title, fontsize=14)
                axs[completed_counter-1].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
            else:
                axs[completed_counter-1, 0].imshow(target_img.squeeze(0).permute(0,1,2))
                axs[completed_counter-1, 0].set_title(title, fontsize=14)
                axs[completed_counter-1, 0].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
                # distractor 
                axs[completed_counter-1, 1].imshow(dist_img.squeeze(0).permute(0,1,2))
                axs[completed_counter-1, 1].set_title("Ground truth: \n" + dist_lbl[0], fontsize=14)
                axs[completed_counter-1, 1].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
       
        if completed_counter == num_batches:
            break

    if batch_size == 1:
        fig.savefig(os.path.join(path, output_path))


if __name__ ==  "__main__":
    # read in cmd args
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--path", help = "path to directory with sandbox and / or full data", nargs = "?", default = "data")
    parser.add_argument("-ri", "--run_inference", help = "flag whether to sample image captions from trained LSTM", action = "store_true", default = False)
    parser.add_argument("-s", "--load_as_sandbox", help = "flag whether to use the sandboxed data (using full dataset otherwise)", action = "store_true", default = False)
    parser.add_argument("-nl", "--num_labels", help = "number of captions per image to use for constructing the dataset (relevant for training as more captions increase the dataset size)", nargs = "?", default = 1, type = int)
    parser.add_argument("-lt", "--labels_type", help = "choose whether long or short captions should be displayed", nargs = "?", default = "long", choices = ["long", "short"])
    parser.add_argument("-b", "--batch_size", help = "batch size (has to be 1 for saving an example plot)", nargs = "?", default = 1, type = int)
    parser.add_argument("-ms", "--max_sequence_length", help = "maximal sequence length to which all encoded ground truth captions are padded / truncated *for training*", nargs = "?", default = 25, type = int)
    parser.add_argument("-v", "--vocab_file", help = "name of vocab file", nargs = "?", default = "vocab.pkl", type = str)
    parser.add_argument("-pr", "--pairs", help = "whether to load pairs of images for contrastive captioning [random or similar] or use single images [none]", nargs = "?", default = "none", type = str, choices = ["random", "similar", "none"])
    parser.add_argument("-idf", "--id_file", help = "file containing image IDs to use for evaluation, e.g., test split of your dataset", nargs = "?", default = None, type = str)
    parser.add_argument("-ft", "--features", help = "list of features along which similar image pairs will be matched", nargs = "?", default = None, type = str)
    parser.add_argument("-pf", "--pairs_file", help = "path to file containing pairs indices", nargs = "?", default = None, type = str)
    parser.add_argument("-fdp", "--full_ds_path", help = "path to directory containing data for loading the full dataset", nargs = "?", default = "data", type = str)
    parser.add_argument("-o", "--output_name", help="name for file where example images with captions will be saved to (if batch size 1)", nargs="?", default = "3dshapes_example.png", type = str)

    args = parser.parse_args()

    main(
        path=args.path, 
        run_inference=args.run_inference, 
        load_as_sandbox=args.load_as_sandbox,
        num_labels=args.num_labels,
        labels_type=args.labels_type,
        batch_size=args.batch_size,
        max_sequence_length=args.max_sequence_length,
        vocab_file=args.vocab_file,
        pairs=args.pairs,
        id_file=args.id_file,
        features=args.features,
        pairs_file=args.pairs_file,  
        full_ds_path=args.full_ds_path,
        output_path=args.output_name,
    )   
    