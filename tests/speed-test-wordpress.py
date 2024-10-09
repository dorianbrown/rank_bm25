
# wget https://huggingface.co/datasets/mteb/cqadupstack-wordpress/resolve/main/corpus.jsonl

import random, json
import time

from rank_bm25 import BM25Okapi

samplenum = 1000
verbose = False
epsilon = 0.00001
print('rank_bm25 speed-test-wordpress\nsamplenum:',samplenum,'| verbose:',verbose,'| epsilon:',epsilon)

# load from jsonl
wcorpus = []
with open('../corpus.jsonl') as f:
  wcstr = f.read()
  wcorpus = wcstr.split('\n')

# create sampled corpus, items, questions
sampledwcorpus = random.sample( wcorpus, samplenum )
items = []
qqs = []
for i in range(0,len(sampledwcorpus)) :
  wjs = json.loads(sampledwcorpus[i])
  #print(i,'---------------',wjs['_id'],'\n',len(wjs['title']),wjs['title'],'\n',len(wjs['text']),wjs['text'])
  items.append( { 'doctext': wjs['text'] } )
  qqs.append( [ wjs['title'], items[-1]['doctext'] ] )

# questions and solutions
random.shuffle(qqs)
questions = [ q[0] for q in qqs ]
questionsolutions = [ q[1] for q in qqs ]


# tokenization function
def mytokenize(s) :
  ltrimchars = ['(','[','{','<','\'','"']
  rtrimchars = ['.', '?', '!', ',', ':', ';', ')', ']', '}', '>','\'','"']
  if type(s) != str : return []
  wl = s.lower().split()
  for i,w in enumerate(wl) :
    if len(w) < 1 : continue
    si = 0
    ei = len(w)
    try :
      while si < ei and w[si] in ltrimchars : si += 1
      while ei > si and w[ei-1] in rtrimchars : ei -= 1
      wl[i] = wl[i][si:ei]
    except Exception as ex:
      print('|',w,'|',ex,'|',wl)
  wl = [ w for w in wl if len(w) > 0 ]
  return wl


# preparing tokenized corpus
tokenized_corpus = [ mytokenize(item['doctext']) for item in items ]

# rank_bm25 and mybm25okapi
rank_bm25_index = BM25Okapi(tokenized_corpus)

stats = { 'get_scores':[], 'get_scores2':[], 'errors':[] }

# Running the questions
for qi,q in enumerate(questions) :
  # tokenize and print question
  tokenizedquestion = mytokenize(q)
  if verbose :
    print('\n----Question',qi,':',q,' | Tokenized: ',tokenizedquestion)
    if questionsolutions and qi<len(questionsolutions) : print('Solution:',questionsolutions[qi])
  
  # get_scores
  starttime = time.time()
  doc_scores = rank_bm25_index.get_scores(tokenizedquestion)
  stats['get_scores'].append((time.time() - starttime))
  
  # get_scores2
  starttime = time.time()
  doc_scores2 = rank_bm25_index.get_scores2(tokenizedquestion)
  stats['get_scores2'].append((time.time() - starttime))
  
  # correctness check
  stats['errors'].append(0)
  for i in range(0,len(doc_scores)) :
    if doc_scores[i] > doc_scores2[i]+epsilon or doc_scores[i] < doc_scores2[i]-epsilon :
      print('!!!! ERROR doc_scores[i] != doc_scores2[i]',i)
      stats['errors'][-1] += 1

  if verbose : print(doc_scores,'\n',doc_scores2)

# print stats
print( 'get_scores time sum : ', sum(stats['get_scores']), 's | get_scores2 time sum : ', sum(stats['get_scores2']),'s | errors sum : ', sum(stats['errors']) )

"""
Example output:

rank_bm25 speed-test-wordpress
samplenum: 1000 | verbose: False | epsilon: 1e-05
get_scores time sum :  1.2493748664855957 s | get_scores2 time sum :  0.7079489231109619 s | errors sum :  0

Process finished with exit code 0

"""
