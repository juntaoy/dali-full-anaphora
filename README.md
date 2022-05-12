# A Cluster Ranking Model for Full Anaphora Resolution
## Introduction
This repository contains code introduced in the following paper:

**[Stay Together: A System for Single and Split-antecedent Anaphora Resolution](https://arxiv.org/abs/2104.05320)**  
Juntao Yu, Nafise Moosavi, Silviu Paun and Massimo Poesio  
In *Proceedings of the 2021 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)*, 2021  
 
**[A Cluster Ranking Model for Full Anaphora Resolution](https://www.aclweb.org/anthology/2020.lrec-1.2/)**  
Juntao Yu, Alexandra Uma and Massimo Poesio  
In *Proceedings of the 12th Language Resources and Evaluation Conference (LREC)*, 2020

An [early version](https://github.com/juntaoy/dali-full-anaphora/tree/naacl2019) of the code also can be used to replicate the results in the following paper:

**[A Crowdsourced Corpus of Multiple Judgments and Disagreement on Anaphoric Interpretation](https://www.google.com/url?q=https%3A%2F%2Faclweb.org%2Fanthology%2Fpapers%2FN%2FN19%2FN19-1176%2F&sa=D&sntz=1&usg=AFQjCNEGeV2V4tsqBI2u4WviKKyxmvm9PQ)**  
Massimo Poesio, Jon Chamberlain, Silviu Paun, Juntao Yu, Alexandra Uma and Udo Kruschwitz  
In *Proceedings of the 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)*, 2019

## Setup Environments
* The code is written in Python 2, the compatibility to Python 3 is not guaranteed.  
* Before starting, you need to install all the required packages listed in the requirment.txt using `pip install -r requirements.txt`.
* After that run `setup.sh` to download the GloVe embeddings that required by the system and compile the Tensorflow custom kernels.

## To use a pre-trained model
* Pre-trained models for our LREC paper can be download from [this link](https://essexuniversity.box.com/s/kak0qsqcxgz7pxsk0vysohwizuem8v67). We provide three pre-trained models:
   * One (best_crac) for CRAC style full anaphora resolution, the model predicts, in addition, the single mentions and the non-referring expressions. 
   * The second model (best_conll) for CoNLL style coreference resolution that only predicts non-singleton clusters.
   * In additional, the third model (best_pd) is trained on the same [Phrase-Detectives-Corpus-2.1.4](https://github.com/dali-ambiguity/Phrase-Detectives-Corpus-2.1.4) as in our NAACL 2019 paper, our latest system has a better scores when compared with our results in the NAACL paper, so we would encourage people to use this model when possible. The model has average CoNLL scores of 75.7% (singletons included) and 66.8% (singletons excluded) and a F1 of 56.7% on detecting non-referring expressions.
   * In the folder you will also find a file called *char_vocab.english.txt* which is the vocabulary file for character-based embeddings used by our pre-trained models.
* Pre-trained models for our NAACL 2021 paper can be download from [this link](https://essexuniversity.box.com/s/1hc0h3yfu6f20046szo8cto7xw0rzdm7). It contains the best model for handling the split antecedent anaphoras (best_split_antecedent).
* Put the downloaded models along with the *char_vocab.english.txt* in the root folder of the code.
* Modifiy the *test_path* and *conll_test_path* accordingly:
   * the *test_path* is the path to *.jsonlines* file, each line of the *.jsonlines* file must in the following format:
   
   ```
  {
  "clusters": [[[0,0],[6,6], [12,12],[22,22],[[2,3],[8,8]], For CoNLL style coreference
  "clusters": [[[0,0,-1],[6,6,-1], [12,12,-1], [22,22,-1]],[[2,3,-1],[8,8,-1]],[[15,15,1]]],[[20,20,-1]],[[24,24,-1]], For CRAC style full anaphora resolution
  "doc_key": "nw",
  "sentences": [["John", "has", "a", "car", "."], ["Did", "He", "washed", "it", "yesteday","?"],["Yes", "he", "did", ",", "it", "was", "sunny", "yesteday", "when", "I", "saw", "him", ",", "we", "even", "had", "a", "good", "chart", "."],
  "speakers": [["sp1", "sp1", "sp1", "sp1", "sp1"], ["sp1", "sp1", "sp1", "sp1", "sp1","sp1"],["sp2","sp2","sp2","sp2","sp2","sp2","sp2","sp2","sp2","sp2","sp2","sp2","sp2","sp2","sp2","sp2","sp2","sp2","sp2","sp2"]]
  "split_antecedents":[[[24,24],[0,0]],[[24,24],[20,20]]], For CRAC style full anahora resolution with split antecedent enabled (optional)
  }
  ```
  
  * For CoNLL style coreference the mentions only contain two properties \[start_index, end_index\] the indices are counted in document level and both inclusive.
  * For CRAC style full anaphora resolution you will need the third property (non-referring types) it starts with `1` and up to `n_types - 1`, in the cases that a mention is a referring mention then put `-1` instead.
  * the *conll_test_path* is the path to the file/folder of gold data in CoNLL or CRAC format respectively.
      * For CoNLL please see the [CoNLL 2012 shared task page](http://conll.cemantix.org/2012/introduction.html) for more detail
      * The CARC 2018 shared task pape can be found [here](http://dali.eecs.qmul.ac.uk/crac18_shared_task)
      * Both CoNLL scorer and the CRAC are included in this repository and is ready to use with our code. Please note we slightly modified the CRAC scorer to interact with our code, the original scorer can be found [here](https://github.com/ns-moosavi/coval)
* Then you need to run the `extract_bert_features.sh` to compute the BERT embeddings for the test set.
* Then use `python evaluate.py config_name` to start your evaluation.

## The extended LEA scorer
* To use the extended LEA scorer for evaluation both single- and split-antecedents you will need export the system output in the same json format mentioned above and using `python lea_naacl2021.py key_json_file sys_json_file split_antecedent_importance (optional)` to get the scores.

## To train your own model
* To train your own model you need first create the character vocabulary by using `python get_char_vocab.py train.jsonlines dev.jsonlines`
* Then you need to run the `extract_bert_features.sh` to compute the BERT embeddings for both training and development sets.
* Finally you can start training by using `python train.py config_name`

## Training speed
The cluster ranking model takes about 16 hours to train (200k steps) on a GTX 1080Ti GPU. 
