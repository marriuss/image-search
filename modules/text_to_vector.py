import numpy as np
from sentence_transformers import SentenceTransformer

# https://www.sbert.net/docs/pretrained_models.html


class TextToVectorModel:

    def __init__(self):
        self._MODEL_NAME = 'all-distilroberta-v1'

    def text2vector(self, text):
        try:
            model = SentenceTransformer(self._MODEL_NAME)
        except Exception as ex:
            print(str(ex))
            return None
        embeddings = model.encode(text, show_progress_bar=False)
        return np.array(embeddings).tolist()
