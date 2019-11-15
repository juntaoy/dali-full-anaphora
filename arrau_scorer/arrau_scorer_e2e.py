import os
import sys

from .arrau import reader
from .arrau import markable
from .eval import evaluator
from .eval.evaluator import evaluate_non_referrings

__author__ = 'ns-moosavi'

def main(key_directory, sys_directory, keep_singletons=True, use_MIN=False, keep_non_referring=False, metric_arg=['muc', 'bcub', 'ceafe'], offical_stdout=True):
    metric_dict = {'lea': evaluator.lea, 'muc': evaluator.muc, 
          'bcub': evaluator.b_cubed, 'ceafe' : evaluator.ceafe}

    if 'all' in metric_arg:
        metrics=[(k, metric_dict[k]) for k in metric_dict]
    else:
        metrics=[]
        for name in metric_dict:
            if name in metric_arg:
                metrics.append((name, metric_dict[name]))

    if len(metrics) == 0:
        metrics=[(name, metric_dict[name]) for name in metric_dict ]

    msg = ""
    if keep_non_referring and keep_singletons:
        msg = "all annotated markables, i.e. including corferent markables, singletons and non-referring markables"
    elif keep_non_referring and not keep_singletons:
        msg = "only coreferent markables and non-referring markables, excluding singletons"
    elif not keep_non_referring and keep_singletons:
        msg = "coreferring markables and singletons, excluding non-referring mentions."
    else:
        msg = "only coreferring markables, excluding singletons and non-referring mentions"

    print('The scorer is evaluating ' , msg,
          ("using the minimum span evaluation setting " if use_MIN else ""))

    conll, nr_f1 = evaluate(key_directory, sys_directory, metrics, keep_singletons, keep_non_referring, use_MIN, offical_stdout)
    return conll, nr_f1

def evaluate(key_directory, sys_directory, metrics, keep_singletons, keep_non_referring, use_MIN, offical_stdout):

    doc_coref_infos, doc_non_referring_infos = reader.get_coref_infos(key_directory, sys_directory, keep_singletons, keep_non_referring, use_MIN)

    conll=0
    conll_subparts_num = 0
    for name, metric in metrics:
        recall, precision, f1 = evaluator.evaluate_documents(doc_coref_infos, metric, beta=1)    
        if name in ["muc", "bcub", "ceafe"]:
            conll+=f1
            conll_subparts_num+=1
        if offical_stdout:
            print(name)
            print('Recall: %.2f'%(recall*100), ' Precision: %.2f'%(precision*100), ' F1: %.2f'%(f1*100))
    if conll_subparts_num ==3:
        conll = (conll/3)*100
        if offical_stdout:
            print('CoNLL score: %.2f'%conll)
    nr_f1 = 0
    if keep_non_referring:
        nr_recall, nr_precision, nr_f1 = evaluate_non_referrings(doc_non_referring_infos)
        if offical_stdout:
            print('Non-referring markable identification scores:')
            print('Recall: %.2f'%(nr_recall*100), ' Precision: %.2f'%(nr_precision*100), ' F1: %.2f'%(nr_f1*100))
    return conll, nr_f1 * 100





