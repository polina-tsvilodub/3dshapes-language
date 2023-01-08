"""
Script for running inference on trained spekaer model, for testing purposes
on the image captioners from the thesis.
"""
import torch
import torch.nn as nn 
from torchvision import transforms
from utils.dataset_utils import get_loader
from utils.DecoderRNN import DecoderRNN
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

def main(path="data", run_inference=False, load_as_sandbox=True):
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
    """

    #### define inference parameters ####
    # has to be 1 if we want to dump a figure with predicted captions, otherwise may be > 1
    batch_size = 1
    # number of steps to run inference for
    num_batches = 10
    # name of trained model
    decoder_file = "pretrained_decoder_3dshapes.pkl"
    # decoder configs
    embed_size = 512
    visual_embed_size = 512
    hidden_size = 512
    # define data loader
    data_loader_test = get_loader(
        load_as_sandbox=load_as_sandbox,
        run_inference=run_inference,
        num_labels=1,
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
    print("VOCAB: ", vocab_size)
    if run_inference:
        decoder = DecoderRNN(embed_size, hidden_size, vocab_size, visual_embed_size)
        # if in train mode, pure sampling runs, in eval mode greedy sampling runs
        decoder.eval()

        # Load the trained weights.
        decoder.load_state_dict(torch.load(os.path.join(path, decoder_file))) 
    
        # Move models to GPU if available.
        decoder.to(device)

    # iterate over nrows image pairs to manually inspect performance of the model
    fig, axs = plt.subplots(nrows=num_batches, ncols=1, figsize=(25, 20))

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

    # for desired number of batches (should match the size of plot), run inference, build plot
    for i, id in enumerate(np.random.choice(len(data_loader_test.dataset), num_batches)):
         
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        new_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=[id])
        data_loader_test.batch_sampler.sampler = new_sampler        
        
        # get the batch
        target_img, target_features, target_lbl, numeric_lbl, target_caption = next(iter(data_loader_test)) 
        
        # print(target_img.shape)
        # print("All short captions for images ID ", id, data_loader_test.dataset.get_labels_for_image(id, caption_type='short'))
        
        if run_inference:
            # move everything to device
            target_caption = target_caption.to(device)
            target_features = target_features.to(device)
            
            # duplicate target since pretrained model expects two images as input
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
            
            cleaned_sentence_t = " ".join(sentence_t.split()[:len_sentence_t])
        
        # show the images, ground truth captions and predicted captions
        if batch_size == 1:
            axs[i].imshow(target_img.squeeze(0).permute(0,1,2))
            if run_inference:
                title = "Ground truth:\n " + target_lbl[0] + "\nPredicted caption:\n " + cleaned_sentence_t
            else:
                title = "Ground truth:\n " + target_lbl[0] 
            axs[i].set_title(title, fontsize=14) 
            axs[i].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
            
    if batch_size == 1:
        fig.savefig(os.path.join(path, "eval_example.png"))


if __name__ ==  "__main__":
    # read in cmd args
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--path", help = "path to directory with sandbox and / or full data", nargs = "?", default = "data")
    parser.add_argument("-ri", "--run_inference", help = "flag whether to sample image captions from trained LSTM", action="store_true", default = False)
    parser.add_argument("-s", "--load_as_sandbox", help = "flag whether to use the sandboxed data (using full dataset otherwise)", action="store_true", default = True)
    
    args = parser.parse_args()

    main(
        path=args.path, 
        run_inference=args.run_inference, 
        load_as_sandbox=args.load_as_sandbox,
    )
    