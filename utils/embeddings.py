import pickle

class Representations:
    def __init__(self, type = 'SBERT', filename = None):
        self.type = type
        self.filename = filename
    
    def get_embeddings(self):

        if (self.type == 'SBERT'):
            emb_sbert_filename= self.filename 
            with open(emb_sbert_filename, 'rb') as f:
                embeddings_sentence = pickle.load(f)

            return embeddings_sentence

        elif ((self.type == 'BERT_mean') | (self.type == 'BERT_max') | (self.type == 'BERT_cls')):
            emb_sentence_filename = self.filename
            with open(emb_sentence_filename, 'rb') as f:
                embeddings_saved = pickle.load(f)
            
            embeddings_sentence = embeddings_saved['embeddings_sentence']

            if (self.type == 'BERT_mean'):
                return embeddings_sentence['mean']
            elif (self.type == 'BERT_max'):
                return embeddings_sentence['max']
            else: return embeddings_sentence['cls']
        
        else:
            print('ERROR: you need to define a valid embedding type from: SBERT , BERT_cls , BERT_mean , BERT_max')
            pass
        