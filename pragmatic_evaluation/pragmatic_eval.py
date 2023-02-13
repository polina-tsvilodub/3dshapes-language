import torch
import os
import numpy as np
import argparse
import pandas as pd
import random
from tqdm import tqdm 
import pickle
import spacy


def compute_pragmatic_scores(
    df_path, 
    annotations_dir,
    output_path,
):
    """
    Function for computing the contrastivity / pragmatic quality of captions produced
    by a model. The expected input comprises as long dataframe containing
    a column with target IDs, a column with distractor IDs, and a column with 
    predicted caption for the target. The function creates a csv reults file with raw
    annotations for each prediction.

    Arguments:
    ---------
    df_path: str
        Path to csv file containing target ids, distractor ids, and traget predictions.
    annotations_dir: str
        Path to directory containing the full dataset files.
    output_path: str
        Path where to write output results csv file to.
    """
    torch.manual_seed(1234)
    random.seed(1234)
    # check that results file exists
    assert os.path.exists(df_path) or df_path is not None, "Please provide the results file with the -rf option or check the spelling of the path you provided!"
    
    # load dependency parser corpus
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        raise ValueError("Dependency parser corpus missing. Please install spacy and run 'python -m spacy download en_core_web_sm' ")
    df = pd.read_csv(df_path)
    # check that the columns exist
    assert all([d in df.columns for d in ["target_id", "prediction", "distractor_id"]]), "The results file must contain the columns target_id, prediction, distractor_id"
    assert os.path.exists(os.path.join(annotations_dir, "3dshapes_labels.npy")), "Make sure to run the sandbox prediction without the flag load_as_sandbox so as to generate the numeric labels numpy file for the full dataset, or double check file name and location."

    # load ground truth numeric annotations
    numeric_labels = np.load(os.path.join(annotations_dir, "3dshapes_labels.npy"))
    # load terminals for constructing admissible captions
    with open(os.path.join(annotations_dir, "3dshapes_grammar_terminals.pkl"), "rb") as f:
        feature_terminals = pickle.load(f)
    
    binary_contrasts = []
    contrast_efficiency_scores = []
    relevance_scores = []
    shape_mentions = []
    scale_mentions = []
    shape_color_mentions = []
    wall_color_mentions = []
    orientation_mentions = []
    floor_color_mentions = []
    shape_mentions_c = []
    scale_mentions_c = []
    shape_color_mentions_c = []
    wall_color_mentions_c = []
    orientation_mentions_c = []
    floor_color_mentions_c = []
    num_n_list = []
    num_d_list = []
    num_f_list = []
    shape_color_mentioned = 0

    for i, r in tqdm(df.iterrows()):
        # retrieve numeric labels of target and distractor
        target_num_label = numeric_labels[r["target_id"]]
        dist_num_label = numeric_labels[r["distractor_id"]]
        # find features along which they differ (i.e., contrastive features)
        contrastive_features_inds = np.where(target_num_label - dist_num_label != 0)[0].tolist()
        # compute the numer of contrastive descriptions
        num_contrastive_descriptions = len(contrastive_features_inds)
        # construct all appropriate description referring to the contrastive features
        # i.e., correct adjective phrases (n-grams) or unigrams, in case the respective feature (relevant for hue only) is unique (e.g., only the object is blue, 
        # then it is assumed to be fine to only say blue, without the noun; but not if e.g. the floor is also blue)

        # construct all sentences for filtering all mentioned features
        ground_truth_expressions = []
        target_shape = feature_terminals["shape"][target_num_label[4]][0].split(" -> ")[-1].replace("'", "")
        colors_list = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink"]
        sizes_list = ["tiny", "small", "medium", "middle sized", "big", "large", "huge", "giant"]

        # for ind, name in list(zip(contrastive_features_inds, contrastive_features_names)):
        for ind, name in enumerate(list(feature_terminals.keys())):
            target_val = target_num_label[ind]
            # retrieve annotation
            terminals = feature_terminals[name][target_val]
            nl_expr = terminals[0].split(" -> ")[-1].replace("'", "")
            # get head noun for color
            head_noun = name.split("_")[0]
            if head_noun == "object":
                # retrieve NL label for the shape
                head_noun = target_shape
                
            elif head_noun == "orientation" or head_noun == "scale" or head_noun == "shape":
                head_noun = ""
            nl_phrase = nl_expr + " " + head_noun
            nl_phrase = nl_phrase.strip()
        
            # check if respective value is unique and we can also consider the unigram (specific to colors, the others are unigrams anyways)
            if head_noun != "orientation" and target_num_label[:3].tolist().count(target_val) == 1:
                ground_truth_expressions.append(nl_expr)
            else:
                ground_truth_expressions.append(nl_phrase.replace("in the ", "").replace("on the ", ""))
            
        contrastive_expressions = [ground_truth_expressions[e] for e in contrastive_features_inds]
    
        # check presence of contrastive expressions in the generated caption
        num_produced_contrasts = 0
        produced_contrasts = []
        produced_contrastive_inds = []
        for j, e in enumerate(contrastive_expressions):
            if e in r["prediction"]:
                num_produced_contrasts += 1
                produced_contrasts.append(e)
                produced_contrastive_inds.append(contrastive_features_inds[j])
                if contrastive_features_inds[j] == 2:
                    shape_color_mentioned = 1
        # print("Number of produced contrastive exprs ", num_produced_contrasts)
        num_d_list.append(num_produced_contrasts)
        
        # actually record produced dicriminative features
        floor_color_mentions_c.append(1 if 0 in produced_contrastive_inds else 0)
        wall_color_mentions_c.append(1 if 1 in produced_contrastive_inds else 0)
        shape_color_mentions_c.append(1 if 2 in produced_contrastive_inds else 0)
        scale_mentions_c.append(1 if 3 in produced_contrastive_inds else 0)
        shape_mentions_c.append(1 if 4 in produced_contrastive_inds else 0)
        orientation_mentions_c.append(1 if 5 in produced_contrastive_inds else 0)

        # compute dependency parse for checking if the shape hue was mentioned (as one phrase)
        dep_parse = nlp(r["prediction"])
        dep_parse_table = pd.DataFrame({
            "token": [t.text for t in dep_parse],
            "tag": [t.tag_ for t in dep_parse],
            "head": [t.head.text for t in dep_parse],
            "dep": [t.dep_ for t in dep_parse],
        })
        
        # retrieve constituents containing nouns (shape, floor, wall, orientation related descriptions)
        heads_tokens = dep_parse_table[(dep_parse_table["tag"] == "NN") | (dep_parse_table["tag"] == "JJ")]["token"].tolist()
        # print("head tokens ", heads_tokens)
        heads_tags = dep_parse_table[(dep_parse_table["tag"] == "NN") | (dep_parse_table["tag"] == "JJ")]["tag"].tolist()
        ngrams = [dep_parse_table[dep_parse_table["dep"] == "ROOT"]["token"].tolist()[0]] 
        num_false_features = 0
        false_toks = []
        for n in heads_tokens:
            if n not in ["picture", "front", "wall", "floor", "unk", "pad", "standing", "a", "on", "in", "the", "of"] and n not in " ".join(ground_truth_expressions):
                num_false_features += 1
                false_toks.append(n)
            
        num_f_list.append(num_false_features)


        # to account for non-discriminative features, check for each ground truth expression if it occurs in the prediction  
        produced_features = [f for f in ground_truth_expressions if f in r["prediction"]]
        num_n_list.append(len(produced_features) - num_produced_contrasts)
        # check occurences of different features for bias statistics
        floor_color_mentions.append(1 if any([n + " floor" in r["prediction"] for n in colors_list ]) else 0)        
        wall_color_mentions.append(1 if any([n + " wall" in r["prediction"] for n in colors_list]) else 0)  
        orientation_mentions.append(1 if any([n in r["prediction"] for n in ["middle", "corner", "left", "right"] ]) else 0)
        shape_mentions.append(1 if any([target_shape in r["prediction"]]) else 0)
        scale_mentions.append(1 if any([a in r["prediction"] for a in sizes_list]) else 0)
        # if no contrastive color was produced, check if it was produced redundantly, either only the adj (if color value unique)
        # or if it cooccurred with the shape
        if shape_color_mentioned == 0:
            target_shape_color = ground_truth_expressions[2]
            if target_num_label[2] != dist_num_label[2]:
                shape_color_mentioned = 1 if target_shape_color.split(" ")[0] in r["prediction"] else 0
            else:
                shape_color_mentioned = 1 if ((target_shape_color.split(" ")[0] in r["prediction"]) and (target_shape in r["prediction"])) else 0
        
        shape_color_mentions.append(shape_color_mentioned)
        binary_contrasts.append(1 if num_produced_contrasts > 0 else 0)
        if num_contrastive_descriptions == 1:
            if num_produced_contrasts == 1:
                contrast_efficiency_scores.append(1)
            else:
                contrast_efficiency_scores.append(0)
        else:
            contrast_efficiency_scores.append(1 - (num_produced_contrasts-1)/(num_contrastive_descriptions-1) if num_produced_contrasts > 0 else 0)


        relevance_scores.append(1-(len(produced_features) - num_produced_contrasts) / (6 - num_contrastive_descriptions))

    df_out = pd.DataFrame({
        "target_id": df["target_id"],
        "distractor_id": df["distractor_id"],
        "prediction": df["prediction"],
        "binary_contrastiveness": binary_contrasts,
        "contrastive_efficiency": contrast_efficiency_scores,
        "relevance": relevance_scores,
        "is_floor_hue": floor_color_mentions,
        "is_wall_hue": wall_color_mentions,
        "is_object_hue": shape_color_mentions,
        "is_scale": scale_mentions,
        "is_shape": shape_mentions,
        "is_orientation": orientation_mentions,
        "is_floor_hue_disc": floor_color_mentions_c,
        "is_wall_hue_disc": wall_color_mentions_c,
        "is_object_hue_disc": shape_color_mentions_c,
        "is_scale_disc": scale_mentions_c,
        "is_shape_disc": shape_mentions_c,
        "is_orientation_disc": orientation_mentions_c,
        "num_mentioned_features": np.array(floor_color_mentions) + np.array(wall_color_mentions) + np.array(shape_color_mentions) + np.array(scale_mentions) + np.array(orientation_mentions) + np.array(shape_mentions),
        "num_nondiscriminative": num_n_list,
        "num_discriminative": num_d_list,
        "num_false": num_f_list,
    })
    df_out.to_csv(output_path)

    # print final statistics to stdout
    print("------------------ Evaluation summary --------------------")
    print("--- Numer of evaluated predictions:              ", len(df_out))
    print("--- Average number of features mentioned in each prediction:         ", df_out["num_mentioned_features"].mean())
    print("--- Average number of contrastive features mentioned:                " , df_out["num_discriminative"].mean())
    print("--- Average number of non-contrastive features mentioned:            ", df_out["num_nondiscriminative"].mean())
    print("--- Average number of false features mentioned:                      ", df_out["num_false"].mean())
    print("--- Average discriminativity:                                        ", df_out["binary_contrastiveness"].mean())
    print("--- Average contrastive efficiency:                                  ", df_out["contrastive_efficiency"].mean())
    print("--- Average relevance:                                               ", df_out["relevance"].mean())
    print("-----------------------------------------------------------")

if __name__ ==  "__main__":
    # read in cmd args
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--path", help = "path to directory with sandbox and / or full data", nargs = "?", default = "../sandbox/data")
    parser.add_argument("-rf", "--results_file", help = "path to file containing predictions and pairs ids to be evaluated", type = str)
    parser.add_argument("-o", "--output_path", help = "path where to write output results to", nargs = "?", default = "pragmatic_eval_results.csv", type = str)
    

    args = parser.parse_args()

    
    compute_pragmatic_scores(
        df_path=args.results_file, 
        annotations_dir=args.path,
        output_path=args.output_path,
    )