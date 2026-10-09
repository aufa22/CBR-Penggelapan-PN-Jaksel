"""Reproducible academic CBR experiment; extracted sentence candidates require review."""
from pathlib import Path
from collections import Counter
import json
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|putusan\.mahkamahagung\.go\.id', ' ', text)
    text = re.sub(r'direktori putusan mahkamah agung republik indonesia|mahkamah agun[a-z]* republik indonesia', ' ', text)
    text = re.sub(r'hal(?:aman)?\s+\d+(?:\s+dari\s+\d+\s+hal\s+putusan)?', ' ', text)
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', ' ', text)).strip()

def sentence_candidate(text):
    # The last mention is a heuristic, not a verified operative ruling.
    starts = list(re.finditer(r'pidana\s+penjara\s+selama\s*', text))
    if not starts:
        return None, '', 'not_found'
    excerpt = text[starts[-1].end():starts[-1].end()+150]
    units = list(re.finditer(r'(\d+)\s*(?:[a-z]+\s+){0,3}?(tahun|bulan)\b', excerpt))
    if not units or units[0].start() > 2:
        return None, excerpt, 'unparsed'
    months = int(units[0].group(1)) * (12 if units[0].group(2) == 'tahun' else 1)
    end = units[0].end()
    if units[0].group(2) == 'tahun' and re.match(r'\s*(?:dan[a-z]*\s+)?\d+\s+(?:satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh)\b', excerpt[end:]) and (len(units)<2 or units[1].group(2)!='bulan'):
        return None, excerpt, 'unparsed'
    if units[0].group(2) == 'tahun' and len(units)>1:
        second = units[1]
        if second.group(2)=='bulan' and re.fullmatch(r'\s*(?:dan[a-z]*\s+)?', excerpt[end:second.start()]):
            months += int(second.group(1));end=second.end()
    return months, excerpt[:end], 'heuristic_requires_review'

def build_case_base(root=ROOT):
    root=Path(root)
    rows=[]
    files=sorted((root/'data/raw').glob('*.txt'))
    if not files:raise ValueError('Tidak ada dokumen TXT dalam data/raw/.')
    for p in files:
        text=clean_text(p.read_text(encoding='utf-8'))
        number=re.search(r'(\d+\s*pid\s+\d+\s+pn\s+jkt\s+sel)',text)
        article=re.search(r'pasal\s+\d+\s+kuhp',text)
        months,excerpt,status=sentence_candidate(text)
        rows.append({'case_id':p.stem,'file_name':p.name,'no_perkara':re.sub(r'(\d)pid', r'\1 pid', number.group(1)) if number else None,
                     'pasal_mention':article.group(0) if article else None,'text_full':text,
                     'candidate_months':months,'sentence_excerpt':excerpt,'label_status':status})
    df=pd.DataFrame(rows)
    for folder in ['data/processed','data/results','data/eval','logs','assets']:(root/folder).mkdir(parents=True,exist_ok=True)
    df.to_csv(root/'data/processed/cases.csv',index=False)
    df.drop(columns='text_full').to_csv(root/'data/processed/extraction_audit.csv',index=False)
    (root/'logs/cleaning.log').write_text(f'Documents processed: {len(df)}\nRaw files preserved.\n',encoding='utf-8')
    return df

class CaseRetriever:
    def __init__(self,cases):
        if cases.empty:raise ValueError('Basis kasus kosong.')
        self.cases=cases.reset_index(drop=True).copy()
        self.vectorizer=TfidfVectorizer(max_features=5000)
        self.matrix=self.vectorizer.fit_transform(self.cases['text_full'])
    def retrieve(self,query,k=5,exclude_case_id=None):
        if not isinstance(k,int) or isinstance(k,bool) or k<1:raise ValueError('k harus bilangan bulat positif.')
        if not isinstance(query,str) or not query.strip():raise ValueError('Query tidak boleh kosong.')
        vector=self.vectorizer.transform([clean_text(query)])
        if vector.nnz==0:
            return self.cases.iloc[:0].assign(similarity=pd.Series(dtype=float))
        scores=cosine_similarity(vector,self.matrix).ravel()
        order=np.argsort(-scores,kind='stable')
        order=[i for i in order if scores[i]>0 and self.cases.iloc[i]['case_id']!=exclude_case_id][:k]
        result=self.cases.iloc[order].copy();result['similarity']=scores[order]
        return result.reset_index(drop=True)
    def reuse(self,query,k=5,exclude_case_id=None):
        top=self.retrieve(query,k,exclude_case_id)
        labelled=top.dropna(subset=['candidate_months'])
        if labelled.empty:return None,top
        votes=Counter(labelled['candidate_months'].astype(int))
        max_votes=max(votes.values())
        # Resolve equal votes by the highest-ranked matching neighbor.
        winner=next(int(v) for v in labelled['candidate_months'] if votes[int(v)]==max_votes)
        return winner,top

def evaluate(retriever,queries,root=ROOT,k=5):
    root=Path(root);rows=[]
    for idx,item in enumerate(queries,1):
        expected=' '.join(item['ground_truth'].lower().split())
        matches=retriever.cases[retriever.cases['no_perkara'].fillna('').str.lower()==expected]
        if len(matches)!=1:raise ValueError(f'Ground truth tidak unik/tidak ditemukan untuk query {idx}.')
        pred,top=retriever.reuse(item['query'],k)
        target_id=matches.iloc[0]['case_id'];ids=top['case_id'].tolist()
        rank=ids.index(target_id)+1 if target_id in ids else None
        rows.append({'query_id':idx,'query':item['query'],'target_case_id':target_id,
                     'rank_at_k':rank,'hit_at_k':int(rank is not None),
                     'reciprocal_rank_at_k':1/rank if rank else 0,
                     'precision_at_k':(1/k if rank else 0),
                     'candidate_prediction_months':pred,
                     'top_case_ids':', '.join(ids),
                     'legacy_actual_solution':item.get('actual_solution')})
    details=pd.DataFrame(rows)
    metrics=pd.DataFrame({'Metric':[f'Hit@{k}',f'Recall@{k}',f'Precision@{k}',f'MRR@{k}'],
                          'Value':[details['hit_at_k'].mean(),details['hit_at_k'].mean(),details['precision_at_k'].mean(),details['reciprocal_rank_at_k'].mean()]})
    details.to_csv(root/'data/results/query_results.csv',index=False)
    metrics.to_csv(root/'data/eval/retrieval_metrics.csv',index=False)
    return metrics,details

def run(root=ROOT):
    cases=build_case_base(root);retriever=CaseRetriever(cases)
    queries=json.loads((Path(root)/'data/eval/queries.json').read_text())
    metrics,details=evaluate(retriever,queries,root)
    pred,top=retriever.reuse(cases.iloc[0]['text_full'],exclude_case_id=cases.iloc[0]['case_id'])
    top.drop(columns='text_full').to_csv(Path(root)/'data/results/retrieval_case_0.csv',index=False)
    print(f'{len(cases)} documents | {len(queries)} queries | {retriever.matrix.shape[1]} TF-IDF features')
    print(metrics.to_string(index=False))
    return cases,retriever,metrics,details

if __name__=='__main__':run()
