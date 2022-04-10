import torch
import warnings
from search_ui import SearchUI
from backend import Backend

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()


def main():
    backend = Backend()
    ui = SearchUI(backend)
    ui.mainloop()


if __name__ == "__main__":
    main()
