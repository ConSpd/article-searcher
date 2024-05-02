import pandas as pd
pd.set_option('display.max_colwidth',None)
selected_rows = 150  # max = 3824
df = pd.read_csv('NewsArticles.csv',nrows = selected_rows, encoding="iso-8859-1", usecols=['title','text'])

import spacy
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm',disable=['tagger','parser','ner'])
tok_text = []


for i,t in tqdm(enumerate(df.text),desc='Clearing empty articles...',total= selected_rows,unit= 'texts'):
    if t != t:
        df = df.drop([i])

for doc in tqdm(nlp.pipe(df.text.str.lower().values),desc='Loading articles...',total= len(df),unit= 'texts'):
    tok = [t.text for t in doc if t.is_alpha]
    tok_text.append(tok)

from rank_bm25 import BM25Okapi
bm25 = BM25Okapi(tok_text)


import time
while True:
    query = input("Search for article: ")
    tokenized_query = query.lower().split(" ")
    t0 = time.time()
    results = bm25.get_top_n(tokenized_query, df.values, n=3)
    t1 = time.time()
    print(f'Searched {selected_rows} records in {round(t1-t0,3)} seconds \n')
    for i,r in enumerate(results):
        print(i+1,":",r[0])
    
    sel = int(input("Select article you want to read "))
    print("\n-",results[sel-1][0],"-\n")
    print(results[sel-1][1])
    print("\n- End of Article -")
