import torch
import numpy as np
from fairseq import utils, tasks
from fairseq import checkpoint_utils
from utils.eval_utils import eval_step
from tasks.mm_tasks.caption import CaptionTask
from torchvision import transforms

# sources:
# https://colab.research.google.com/drive/1Q4eNhhhLcgOP4hHqwZwU1ijOlabgve1W?usp=sharing
# https://github.com/OFA-Sys/OFA


class ImageCaptionModel:

    def __init__(self, checkpoint_path):
        self.CHECKPOINT_PATH = checkpoint_path

        tasks.register_task('caption', CaptionTask)
        self._USE_CUDA = torch.cuda.is_available()
        self._UDE_FP16 = True

        overrides = {"bpe_dir": "utils/BPE",
                     "eval_cider": False,
                     "beam": 5,
                     "max_len_b": 16,
                     "no_repeat_ngram_size": 3,
                     "seed": 7}

        self.MODELS, self.CGF, self.TASK = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(self.CHECKPOINT_PATH),
            arg_overrides=overrides
        )

        for model in self.MODELS:
            model.eval()
            if self._UDE_FP16:
                model.half()
            if self._USE_CUDA and not self.CGF.distributed_training.pipeline_model_parallel:
                model.cuda()
            model.prepare_for_inference_(self.CGF)

        self._GENERATOR = self.TASK.build_generator(self.MODELS, self.CGF.generation)

    def _encode_text(self, text, length=None, append_bos=False, append_eos=False):
        s = self.TASK.tgt_dict.encode_line(
            line=self.TASK.bpe.encode(text),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        bos_item = torch.LongTensor([self.TASK.src_dict.bos()])
        eos_item = torch.LongTensor([self.TASK.src_dict.eos()])
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([bos_item, s])
        if append_eos:
            s = torch.cat([s, eos_item])
        return s

    def _construct_sample(self, image):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        patch_resize_transform = transforms.Compose([
            lambda img: img.convert("RGB"),
            transforms.Resize((self.CGF.task.patch_image_size,
                               self.CGF.task.patch_image_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        pad_idx = self.TASK.src_dict.pad()
        patch_image = patch_resize_transform(image).unsqueeze(0)
        patch_mask = torch.tensor([True])
        src_text = self._encode_text(" what does the image describe?",
                                     append_bos=True,
                                     append_eos=True).unsqueeze(0)
        src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
        sample = {
            "id": np.array(['42']),
            "net_input": {
                "src_tokens": src_text,
                "src_lengths": src_length,
                "patch_images": patch_image,
                "patch_masks": patch_mask
            }
        }
        return sample

    @staticmethod
    def _apply_half(t):
        if t.dtype is torch.float32:
            return t.to(dtype=torch.half)
        return t

    def create_caption(self, image):
        sample = self._construct_sample(image)
        sample = utils.move_to_cuda(sample) if self._USE_CUDA else sample
        sample = utils.apply_to_sample(self._apply_half, sample) if self._UDE_FP16 else sample
        with torch.no_grad():
            result, scores = eval_step(self.TASK, self._GENERATOR, self.MODELS, sample)
        caption = result[0]["caption"]
        return caption
