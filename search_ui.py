import os
import pickle
import shutil
import tkinter as tk
from PIL import ImageTk, Image

DATASET_PATH = "dataset"
RESULTS_PATH = "search_results"

GALLERY_SIZE = (800, 600)


class Gallery(tk.Toplevel):

    def __init__(self, master, results: list):
        super(Gallery, self).__init__(master=master)
        assert (isinstance(results, list))
        w, h = GALLERY_SIZE
        self.geometry(f"{w}x{h}")
        self.resizable(width=False, height=False)
        self._results = results
        self._current_index = 0
        self._init_graphical_elements()

    def _init_graphical_elements(self):
        self._canvas = tk.Canvas(self)
        self._canvas.grid(row=0, column=0, columnspan=3)
        self._caption_label = tk.Label(self, text="caption")
        self._caption_label.grid(row=1, column=0)
        self._index_label = tk.Label(self, text="0")
        self._index_label.grid(row=2, column=0)
        self._display_image(0)
        tk.Button(self, text='Previous image', command=lambda: self._change_image(-1)) \
            .grid(row=3, column=0, sticky="SW")
        tk.Button(self, text='Next image', command=lambda: self._change_image(1)) \
            .grid(row=3, column=1, sticky="SE")

    def _change_image(self, delta):
        self._current_index += delta
        length = len(self._results)
        left_border = -length
        right_border = length - 1
        if self._current_index < left_border:
            self._current_index = right_border
        elif self._current_index > right_border:
            self._current_index = left_border
        self._display_image(self._current_index)

    def _display_image(self, index):
        result = self._results[index]
        image = result["image"]
        caption = result["caption"]
        score = result["score"]
        self._index_label.text = f"Image index: {index + 1}"
        self._index_label.update()
        self._caption_label.text = caption
        self._caption_label.update()
        self._canvas.delete("all")
        self._canvas.create_image(20, 20, anchor="s", image=image)
        self._canvas.image = image


class SearchUI(tk.Tk):

    def __init__(self, backend=None):
        super(SearchUI, self).__init__()
        self.WIDTH = 350
        self.HEIGHT = 100
        self.title("Images search")
        self.geometry(f"{self.WIDTH}x{self.HEIGHT}")
        self.resizable(width=False, height=False)
        self._backend = backend
        self._init_grapical_elements()

    def store_dataset(self):
        result = self._backend.try_store_dataset(DATASET_PATH)
        if len(result) > 0:
            self.save_not_stored(DATASET_PATH, result)

    def mainloop(self, n: int = ...) -> None:
        super(SearchUI, self).mainloop()
        self._clear_directory(RESULTS_PATH)

    def _init_grapical_elements(self):
        self._temporary_object = None
        self._gallery = None
        frame_width = self.WIDTH * 0.9
        self._main_frame = tk.Frame(master=self, width=frame_width)
        self._main_frame.pack()
        query = tk.StringVar()
        size = tk.IntVar()
        tk.Label(master=self._main_frame, text="Input query:", anchor="center").grid(row=0, column=0)
        tk.Label(master=self._main_frame, text="Input size:", anchor="center").grid(row=0, column=1)
        tk.Entry(master=self._main_frame, textvariable=query).grid(row=1, column=0)
        tk.Spinbox(master=self._main_frame, from_=1, to=30, textvariable=size).grid(row=1, column=1)
        tk.Button(master=self._main_frame,
                  text="Search",
                  anchor="center",
                  command=lambda: self._init_search(query.get(), size.get())) \
            .grid(row=2, column=0, columnspan=2)

    def _init_search(self, query, size):
        self._destroy_object(self._temporary_object)
        self._destroy_object(self._gallery)
        if query != "":
            results = self._backend.search_images_by_text(query, size)
            if len(results) > 0:
                self._results_to_os(query, results)
                gallery_button = tk.Button(master=self._main_frame,
                                           text="Open results gallery",
                                           anchor="center",
                                           command=lambda: self._results_to_gallery(results))
                gallery_button.grid(row=3, column=0, columnspan=2)
                self._temporary_object = gallery_button
                return
            else:
                error_text = "No images found."
        else:
            error_text = "Query is empty."
        error_label = tk.Label(master=self._main_frame, text=error_text, anchor="center")
        error_label.grid(row=3, column=0, columnspan=2)
        self._temporary_object = error_label

    def _results_to_os(self, query, results):
        query_results_path = os.path.join(RESULTS_PATH, query)
        if os.path.isdir(query_results_path):
            self._clear_directory(query_results_path)
        else:
            os.mkdir(query_results_path)
        for r in results:
            image_name = r["name"]
            image_path = os.path.join(DATASET_PATH, image_name)
            score = f"{r['score']:.4}"
            _, ext = os.path.splitext(image_name)
            caption = r["caption"]
            new_name = "".join([score, " ", caption, ext])
            new_path = os.path.join(query_results_path, new_name)
            shutil.copy2(image_path, new_path)
        self._open_directory(query_results_path)

    def _results_to_gallery(self, results):
        if self._gallery is None:
            for r in results:
                image_name = r["name"]
                image_path = os.path.join(DATASET_PATH, image_name)
                image = self._backend.read_image(image_path)
                image = self._backend.resize_image(image, GALLERY_SIZE)
                r["image"] = ImageTk.PhotoImage(image)
                r.pop("name")
        self._gallery = Gallery(master=self, results=results)
        self._gallery.mainloop()

    @staticmethod
    def _destroy_object(gui_object):
        if gui_object is not None:
            gui_object.destroy()
            gui_object = None

    @staticmethod
    def _open_directory(directory_path):
        os.startfile(directory_path)

    @staticmethod
    def _clear_directory(directory_path):
        if not os.path.isdir(directory_path):
            return
        for root, dirs, files in os.walk(directory_path):
            for f in files:
                try:
                    os.unlink(os.path.join(root, f))
                except Exception as ex:
                    print(str(ex))
            for d in dirs:
                try:
                    shutil.rmtree(os.path.join(root, d))
                except Exception as ex:
                    print(str(ex))

    @staticmethod
    def save_not_stored(dataset_path, images_paths):
        new_path = os.path.join(dataset_path, "not_stored")
        if not os.path.isdir(new_path):
            os.mkdir(new_path)
        for path in images_paths:
            shutil.move(path, new_path)
