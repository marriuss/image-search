import os
import cv2
from PIL import Image
from modules import ImageCaptionModel, TextToVectorModel, ESDataStoring, ESDataSearch

CHECKPOINT_PATH = "checkpoints/caption.pt"


class Backend:

    def __init__(self):
        self._ic_model = ImageCaptionModel(CHECKPOINT_PATH)
        self._t2v_model = TextToVectorModel()

    def try_store_image(self, image_path):
        _, image_name = os.path.split(image_path)
        caption, vector = self._image2vector(image_path)
        es_connection = ESDataStoring()
        result = es_connection.try_add_document({"name": image_name, "caption": caption, "vector": vector})
        es_connection.close_connection()
        return result

    def try_store_dataset(self, dataset_path):
        images = [i for i in os.listdir(dataset_path) if i.endswith((".jpg", ".jpeg"))]
        not_stored = []
        for image_name in images:
            image_path = os.path.join(dataset_path, image_name)
            result = self.try_store_image(image_path)
            if not result:
                not_stored.append(image_path)
        return not_stored

    def search_images_by_text(self, text, size):
        vector = self._t2v_model.text2vector(text)
        results = self._search_vector(vector, size)
        return results

    def _create_image_caption(self, image):
        caption = self._ic_model.create_caption(image)
        return caption

    def _text2vector(self, text):
        vector = self._t2v_model.text2vector(text)
        return vector

    def _image2vector(self, image_path):
        image = self.read_image(image_path)
        caption = self._create_image_caption(image)
        vector = self._text2vector(caption)
        return caption, vector

    @staticmethod
    def _search_vector(vector, size):
        es_connection = ESDataSearch()
        results = es_connection.search(vector, size)
        es_connection.close_connection()
        return results

    @staticmethod
    def read_image(image_path):
        return Image.open(image_path)

    @staticmethod
    def resize_image(image, max_size):
        width, height = image.size
        max_w, max_h = max_size
        scale = min(width / max_w, height / max_h)
        if scale > 1:
            image = cv2.resize(image, dsize=(int(width * scale / 100), int(height * scale / 100)))
        return image
