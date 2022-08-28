## 3dshapes-language 

This repo contains materials for generating an annotations dataset for the 3Dshapes image dataset (Burgess & Kim, 2018).
These annotations are in English and were created for the supplied vector labels for the M.Sc. thesis "Language Drift of Multi-Agent Communication Systems in Reference Games" submitted to the University of Osnabrück. 

The contents of the repository are described below.

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


### References

Burgess, C., & Kim, H. (2018). 3d shapes dataset