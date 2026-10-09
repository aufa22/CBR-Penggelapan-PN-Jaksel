import hashlib
import unittest
from src.cbr import ROOT, build_case_base, CaseRetriever, sentence_candidate

class CBRTests(unittest.TestCase):
    def test_duration_units(self):
        for text, expected in [('pidana penjara selama 2 dua tahun',24),('pidana penjara selama 2 tahun dan 6 bulan',30),('pidana penjara selama 1 satu tahun 8 delapan bulan',20),('pidana penjara selama 10 sepuluh bulan',10)]:
            self.assertEqual(sentence_candidate(text)[0],expected)
    def test_last_mention(self):
        self.assertEqual(sentence_candidate('pidana penjara selama 2 dua tahun tuntutan kemudian pidana penjara selama 8 delapan bulan')[0],8)
    def test_unreadable(self):
        self.assertIsNone(sentence_candidate('pidana penjara selama tahun 3 menetapkan')[0])
        self.assertIsNone(sentence_candidate('pidana penjara selama 1 satu tahun dan 2 dua bnulan')[0])
    def test_preserve_sources_and_retrieval(self):
        paths=sorted((ROOT/'data/raw').glob('*.txt'))
        before=[hashlib.sha256(p.read_bytes()).digest() for p in paths]
        cases=build_case_base()
        self.assertEqual(before,[hashlib.sha256(p.read_bytes()).digest() for p in paths])
        self.assertEqual(len(cases),30)
        retriever=CaseRetriever(cases)
        query=cases.iloc[0]['text_full']
        top=retriever.retrieve(query,5,exclude_case_id=cases.iloc[0]['case_id'])
        self.assertNotIn(cases.iloc[0]['case_id'],top['case_id'].tolist())
        self.assertTrue(top['similarity'].is_monotonic_decreasing)
        pred,unknown=retriever.reuse('zzzzxxyyqq')
        self.assertIsNone(pred);self.assertTrue(unknown.empty)
        for query,k in [('',5),('uang',0),('uang',True)]:
            with self.assertRaises(ValueError):retriever.retrieve(query,k)
    def test_vote_tie_uses_ranking(self):
        import pandas as pd
        r=CaseRetriever(pd.DataFrame({'case_id':['a','b'],'text_full':['uang perusahaan','dana korban'],'candidate_months':[12,24]}))
        r.retrieve=lambda *args,**kwargs:pd.DataFrame({'candidate_months':[24,12]})
        self.assertEqual(r.reuse('uang')[0],24)

if __name__=='__main__':unittest.main()
