## Introduction

A new version of the system which has a better performance can be find [here](https://github.com/juntaoy/dali-full-anaphora)

This repository contains code introduced in the following paper:

**A Crowdsourced Corpus of Multiple Judgments and Disagreement on Anaphoric Interpretation**
Massimo Poesio, Jon Chamberlain, Silviu Paun, Juntao Yu, Alexandra Uma and Udo Kruschwitz  
In *Proceedings of the 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)*, 2019

## Setup Environments
* The code is written in Python 2, the compatibility to Python 3 is not guaranteed.  
* Before starting, you need to install all the required packages listed in the requirment.txt using `pip install -r requirements.txt`.
* After that run `setup.sh` to download the GloVe embeddings that required by the system and compile the Tensorflow custom kernels.

## To use a pre-trained model
* Pre-trained models can be download from [this link](https://essexuniversity.box.com/s/z2c250pcoal157bo6w940r1rvpjxx3el). We provide our pre-trained model (naacl_2019_pd) reported in our paper:
   * In the folder you will also find a file called *char_vocab.english_pd2.0.txt* which is the vocabulary file for character-based embeddings used by our pre-trained models.
* Put the downloaded models along with the *char_vocab.english_pd2.0.txt* in the root folder of the code.
* Modifiy the *test_path* and *conll_test_path* accordingly:
   * the *test_path* is the path to *.jsonlines* file, each line of the *.jsonlines* file must in the following format:
   
   ```
  {
  "clusters": [[[0,0],[4,4]],[[2,3],[7,8]], For CoNLL style coreference
  "clusters": [[[0,0,-1],[4,4,-1]],[[2,3,-1],[7,8,-1],[[13,13,1]]], For CRAC style full anaphora resolution
  "doc_key": "nw",
  "sentences": [["John", "has", "a", "car", "."], ["He", "washed", "the", "car", "yesteday","."],["Really","?","it", "was", "raining","yesteday","!"]],
  "speakers": [["sp1", "sp1", "sp1", "sp1", "sp1"], ["sp1", "sp1", "sp1", "sp1", "sp1","sp1"],["sp2","sp2","sp2","sp2","sp2","sp2","sp2"]]
  }
  ```
  
  * For CoNLL style coreference the mentions only contain two properties \[start_index, end_index\] the indices are counted in document level and both inclusive.
  * For CRAC style full anaphora resolution you will need the third property (non-referring types) it starts with `1` and up to `n_types - 1`, in the cases that a mention is a referring mention then put `-1` instead.
  * the *conll_test_path* is the path to the file/folder of gold data in CoNLL or CRAC format respectively.
      * For CoNLL please see the [CoNLL 2012 shared task page](http://conll.cemantix.org/2012/introduction.html) for more detail
      * The CARC 2018 shared task pape can be found [here](http://dali.eecs.qmul.ac.uk/crac18_shared_task)
      * Both CoNLL scorer and the CRAC are included in this repository and is ready to use with our code. Please note we slightly modified the CRAC scorer to interact with our code, the original scorer can be found [here](https://github.com/ns-moosavi/coval)
* Then use `python evaluate.py config_name` to start your evaluation

## To train your own model
* To train your own model you need first create the character vocabulary by using `python get_char_vocab.py train.jsonlines dev.jsonlines`
* Then you need to run the `python cache_elmo.py train.jsonlines dev.jsonlines` to store the ELMo embeddings in the disk, this will speed up your training a lot.
* Finally you can start training by using `python train.py config_name`

## Training speed
The model takes about 3 days to train (200k steps) on a GTX 1080Ti GPU.
