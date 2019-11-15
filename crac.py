import codecs,os
from arrau_scorer.arrau_scorer_e2e import main

def output_crac(gold_path, predictions,prediction_path):
  for doc_key, marks in predictions.items():
    output_path = doc_key[doc_key.rfind('/')+1:]+'.CONLL'
    input_path = os.path.join(gold_path,output_path)
    tokens = []
    skip1stline = False
    for line in codecs.open(input_path,'rb','utf-8'):
      if not skip1stline:
        skip1stline=True
        continue
      if len(line) > 0:
        tokens.append(line.split()[0])
    marks = sorted(marks)
    output_path = os.path.join(prediction_path,output_path)
    with codecs.open(output_path,'wb','utf-8') as output_file:
      output_file.write('TOKEN\tMARKABLE\tREFERENCE\n')
      for tid, tok in enumerate(tokens):
        m, ref =[],[]
        for mid, (s,e,sid,type) in enumerate(marks):
          l = ''
          if tid == s:
            l='B-'
          elif s< tid <= e:
            l='I-'
          if len(l)>0:
            m.append('%smarkable_%d=set_%d' % (l,mid,sid))
            ref.append(type)
        m = '@'.join(reversed(m))
        ref = '@'.join(reversed(ref))
        output_file.write('%s\t%s\t%s\n'%(tok, m, ref))
      output_file.close()



def eval_crac(gold_path, predictions, prediction_path, official_stdout=False):
  output_crac(gold_path,predictions,prediction_path)
  conll, nr_f1 = main(gold_path, prediction_path, keep_singletons=True, keep_non_referring=True,
                      offical_stdout=official_stdout)
  avg_f1 = conll * 0.85 + nr_f1*0.15
  return avg_f1,conll,nr_f1
