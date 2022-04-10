from elasticsearch import Elasticsearch
from elasticsearch_dsl import Index, Document, Text, DenseVector

INDEX_NAME = "images"


class Image(Document):
    name = Text(required=True)
    caption = Text(required=True)
    vector = DenseVector(dims=768, required=True)


class ESConnection:

    def __init__(self):
        self._es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
        self._INDEX = Index(INDEX_NAME)

        if self._connected():
            if not self._INDEX.exists(using=self._es):
                self._INDEX.document(Image)
                self._INDEX.create(using=self._es)

    def close_connection(self):
        if self._connected():
            self._es.transport.close()

    def _connected(self):
        return self._es.ping()


class ESDataStoring(ESConnection):

    def __init__(self):
        super(ESDataStoring, self).__init__()

    def try_add_document(self, image_data):
        if not self._connected():
            print("No Elasticsearch connection.")
            return
        result = False
        try:
            image_document = Image(**image_data)
            image_document.save(using=self._es, index=INDEX_NAME)
            result = True
        except Exception as ex:
            print(ex.__class__, str(ex), sep=":")
        return result


class ESDataSearch(ESConnection):

    def __init__(self):
        super(ESDataSearch, self).__init__()

    def search(self, vector, size):
        if not self._connected():
            return None
        results = None
        query = self._form_search_query(vector)
        try:
            response = self._es.search(body=query, index=INDEX_NAME, size=size)["hits"]["hits"]
            results = []
            for r in response:
                source = r["_source"]
                source.pop("vector")
                results.append({**source, "score": r["_score"]})
        except Exception as ex:
            print(str(ex))
        return results

    @staticmethod
    def _form_search_query(vector):
        return {
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                        "params": {
                            "query_vector": vector
                        }
                    }
                }
            }
        }
