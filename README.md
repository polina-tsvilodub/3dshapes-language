## 3dshapes-language 

This repo contains materials for generating an annotations dataset for the 3Dshapes image dataset (Burgess & Kim, 2018) and using it for inference, training and evaluation of neural (contrastive) image captioners.
These annotations are in English and were created for the supplied vector labels of the [3Dshapes dataset](https://github.com/deepmind/3d-shapes) (Burgess & Kim, 2018) for the M.Sc. thesis "Language Drift of Multi-Agent Communication Systems in Reference Games" submitted to the University of Osnabrück. 

The contents of the repository are described below. Generally, the `language` directory contains materials necessary for generating captions from scratch. For using provided captions, please skip to the sections `pragamtic_evaluation` and `sandbox`. For instructions on using both the reduced sandbox version of the dataset and the full dataset, see the `sandbox` section. 

## `language`

### `data`

This directory is the target for dumping the generated caption files. 
Additionally, it contains the grammar files used for generating the annotations;

* `exhaustive_grammar.txt`: production rules for creating exhaustive natural language descriptions given vector labels of images. That is, created sententences describe all six features of the image. Multiple synonymous grammatical constructions can be generated for each label.
* `short_sentences_grammar.txt`: production rules for creating short natural language descriptions given vector labels of images. That is, created sententences describe up to three out of six features of the image. Multiple synonymous grammatical constructions can be generated for each label.
* `singleton_rules_grammar_exh.txt`: an extension of `exhaustive_grammar.txt` created for parsing sample sentences with these rules with `nltk`. 
* `singleton_rules_grammar_short.txt`: an extension of `short_sentences_grammar.txt` created for parsing sample sentences with these rules with `nltk`. 

### `src`

The note book contains code for:

* creating exhaustive and short natural language captions from vector labels
* parsing sentences with the hand-crafted grammars for closer analysis of the syntactic structures
* plotting the parse distributions
* testing embedding 3Dshapes images with a pretrained ResNet50 instance
* exploring syntactic structure of the MS COCO dataset samples for comparison.

## `pragmatic_evaluation`

This directory contains the script which allows to evaluate pragmatic quality of image captions from the perspective of a referential task where the captions ought to be contrastive (i.e., set apart the target from the distractor), but not overinformative. The evaluation can be run on image pairs (i.e., one target and one distractor). 

* `pragmatic_eval.py`: The script expects a csv file containing the columns "prediction", "target_id", "distractor_id" where the last two should contain numeric image IDs. It writes an evaluation file to a separate output file, appending the following columns with metrics reported in the paper and raw data used for their computation to the original dataframe: "binary_contrastiveness" (discriminativity), "relevance", "contrastive_efficiency", "is_<feature_name>" (indicator columns indicating whether the respective feature was mentioned in the prediction), "is_<feature_name>_disc" (indicator column indicating whether the respective feature was contrastive for the given pair, if it was mentioned), "num_nondiscriminative" (number of non-contrastive features mentioned in the prediction), "num_discriminative" (number of contrastive features mentioned in the prediction), "num_false" (number of features mentioned in the prediction which are false of the target), "num_mentioned_features" (total number of features mentioned in the prediction).
Additionally, summary statistics are printed to the console.
To run the evaluation, navigate to this directory and run `python pragmatic_eval.py -rf="results_file.csv" -o="output_file.csv"`, substituting the paths with your respective results file and desired output file location. Please set the full dataset location with the argument `-p=<you location>` (default is `-p='../sandbox/data'`).

## `sandbox`

This directory contains a sandbox version of the 3Dshapes dataset and the respective annotations (their creation is described above) which can be accessed directly in this repository. The sandbox contains 1000 images sampled at random from the full dataset. The IDs of the sampled images can be inspected in `sandbox/data/sandox_IDs_3dshapes_1000.txt`. Below, the subdirectories are documented in more detail. The files with "_similarPairs" in their name are extended versions of the scripts which can be used for accessing different kinds of pairs of images on the full dataset (see below for full dataset access instructions).

The sandbox dataset is provided alongside with an entrypoint script `sandbox/img2text_inference.py` and a dataloader which allows for the following: 

* loading the sandbox dataset and displaying example images with captions
    * in order to do this, clone this repository. Please do NOT change the names of the files in `sandbox/data`. Navigate to `sandbox`. Run `python img2text_inference.py`.
* loading the sandbox dataset and displaying example images with captions and predictions from an image captioner LSTM model trained by Polina Tsvilodub (polina.tsvilodub@uni-tuebingen.de). This model uses a ResNet-50 visual encoder backbone, such that the sandbox contains precomputed ResNet image features.
    * in order to do this, clone this repository. Please do NOT change the names of the files in `sandbox/data`. Navigate to `sandbox`. Set the flag `run_inference=True` in the last line of `sandbox/img2text_inference.py`. Run `python img2text_inference.py`.
* loading the full 3Dshapes dataset. **WARNING**: this will require large downloads and will create large files on your hard drive (will require app. 13GB space).
    * in order to do this, you will need to *manually* download the following files into the `sandbox/data` directory. **IMPORTANT**: do NOT modify the names of the downloaded files.
        * the full set of ResNet image features [here](https://drive.google.com/file/d/1OZ7a2xoMK9uF5akMpEo7fQee3J3QeQ3F/view?usp=share_link)
        * the long captions file [here](https://drive.google.com/file/d/1lwxmF9FGteoSZ4dA483bOmItqlEFVLMZ/view?usp=share_link)
        * the short captions file [here](https://drive.google.com/file/d/1rVk7b6IZ5unR-Oihjpf81xR-SCaRHEg_/view?usp=share_link)
        * the original 3Dshapes dataset, following the instruction [here](https://github.com/deepmind/3d-shapes)
    * navigate to `sandbox`. Set the flag `load_as_sandbox=False` in the last line of `sandbox/img2text_inference.py`. Run `python img2text_inference.py`.

For working with pairs of images, possibly retrieved matched on certain features, the script `sandbox/img2text_inference_similarPairs.py` is provided. It's general structure is the same as for the script described above. Details on extended functionality are below.
* to load pairs of images combined at random from the sandbox, run `python img2text_inference_similarPairs.py -s -pr="random"`.
* to load the sandbox dataset and displaying example images with captions and predictions the provided image captioner LSTM model run `python img2text_inference_similarPairs.py -s -ri`.
* to inspect short annotations of the images, set the argument `-lt="short"` additionally to the call.
* to use the full dataset and create similar image pairs (possible *only* on the full dataset), make sure to download the full dataset files as described above into the directory `sandbox/data`. Once completed, navigate to `sandbox`. Run `python img2text_inference_similarPairs.py`.
    * with the full dataset, additionally to "random" pairs (`pr="random"`) and single images (default without `-pr` arg), "similar" image pairs can be retrieved. To do so, run `python img2text_inference_similarPairs.py -pr="similar"`. By default, image pairs will be matched on three random features. If the user wants them to match on specific features, run `python img2text_inference_similarPairs.py -pr="similar" -ft="object_hue,orientation"` (example where the features between the target and distractor will be orientation and object color).
    * a file with selected image indices from which images may be sampled or a file containing a list of given image pair IDs can be used for loading when the full dataset is available. To do so, use the `-idf` arg for the former and `-pf` arg for the latter.

### `data`

* `pretrained_decoder_3dshapes.pkl`: weights of the trained LSTM image captioner. Contact the author if you have any questions.
* `sandbox_3Dshapes_1000.pkl`: sandbox split of the 3Dshapes dataset. The pickle file compresses a dictionary which was the following keys and values. 
    * `original_ids`: a list of original image IDs of the dataset subsample. These can also be found in `sandbox/data/sandox_IDs_3dshapes_1000.txt`.
    * `images`: a list of length 1000 containing the raw images in np.ndarray format of shape (64, 64, 3)
    * `labels_numeric`: a list of length 1000 containing the raw labels in np.ndarray format of shape (6,) (see [original repository](https://github.com/deepmind/3d-shapes) for explanations of the annotations)
    * `labels_long`: a nested list of length 1000, each sublist of length 20, containing string long captions for the respective images
    * `labels_short`: a nested list of length 1000, each sublist of length 27, containing string short captions for the respective images
* `sandbox_3Dshapes_resnet50_features_1000.pt`: torch.Tensor of shape (1000, 2048) containing ResNet features of the sandbox images
* `sandox_IDs_3dshapes_1000.txt`: list of original IDs of the images used in the sandbox (order preserved as in the sandbox pickle file)
* `vocab.pkl`: vocab file for mapping strings on to token IDs for performing inference with the trained model
* `3dshapes_grammar_terminal.pkl`: compressed dictionary containing the terminal rules of the grammar which was used for generating the captions, providing the mapping of raw numeric image feature values (i.e., numeric labels for object color, shape, scale, orientation, wall color and floor color) to tthe natural language expressions with which they are described in the dataset. Used for pragmatic evaluation.
* `categories2imgIDs_3dshapes.json`: a dictionary mapping image feature values to all image IDs from the full dataset which contain that given feature value. It is used for loading similar image pairs (see above; only available in the full dataset loading regime).

Please place all files you download for accessing the full dataset as described above in this directory. Please do not modify the file names. If you move the entire data directory, make sure to adjust the path in the entrypoint script `sandbox/img2text_inference.py` or `sandbox/img2text_inference_similarPairs.py`.

### `utils`

* `dataset_utils.py`: script containing the custom instances of PyTorch Dataset and DataLoader classes for loading the annotated 3Dshapes dataset.
* `dataset_utils_similarPairs.py`: script containing the custom instances of PyTorch Dataset and DataLoader classes for loading pairs of images and their respective annotations, possibly matched by provided features, from the annotated 3Dshapes dataset.
* `DecoderRNN.py`: script defining the LSTM image captioner for which the weights are provided in this repository.

# References

Burgess, C., & Kim, H. (2018). 3d shapes dataset