import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
import numpy as np
import os, random, time, glob, argparse, yaml
from typing import Dict
from fairseq import utils, tasks
from fairseq import checkpoint_utils
from utils.eval_utils import eval_step, eval_caption
from captioning_tasks.mm_tasks.caption import CaptionTask
from model.ofa import OFAModel
from PIL import Image, ImageDraw, ImageFont

class ImgCaptionInferencer:
    def __init__(self, config) -> None:
        self.__use_cuda = config["cuda"]
        self.__fp16 = config["fp16"]
        self.__weight_path = config["weight"]
        self.__overrides = {"eval_cider": config["eval_cider"], 
                            "beam": config["beam"], 
                            "max_len_b": config["max_len_b"], 
                            "no_repeat_ngram_size": config["no_repeat_ngram_size"], 
                            "seed": config["seed"]}

        self.models, self.generator, self.task, self.cfg = self.__build_model()

    def __build_model(self):
        tasks.register_task('caption', CaptionTask)
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(self.__weight_path),
            arg_overrides=self.__overrides
        )

        # Move models to GPU
        for model in models:
            model.eval()
            if self.__fp16:
                model.half()

            if self.__use_cuda and not cfg.distributed_training.pipeline_model_parallel:
                model.cuda()
            model.prepare_for_inference_(cfg)
        
        # Initialize generator
        generator = task.build_generator(models, cfg.generation)

        return models, generator, task, cfg
    
    def __apply_half(self, t):
        if t.dtype is torch.float32:
            return t.to(dtype=torch.half)
        return t

    def __text_preprocess(self):
        bos_item = torch.LongTensor([self.task.src_dict.bos()])
        eos_item = torch.LongTensor([self.task.src_dict.eos()])
        pad_idx = self.task.src_dict.pad()
        return bos_item, eos_item, pad_idx

    def __encode_text(self, text, bos_item, eos_item, length=None, append_bos=False, append_eos=False):
        s = self.task.tgt_dict.encode_line(
            line = self.task.bpe.encode(text),
            add_if_not_exist=False,
            append_eos=False
        ).long()

        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([bos_item, s])
        if append_eos:
            s = torch.cat([s, eos_item])
        return s
    
    def __construct_sample_image(self, image: Image):
        # mean = [0.5, 0.5, 0.5]
        # std = [0.5, 0.5, 0.5]

        patch_resize_transform = transforms.Compose(
            [
                lambda image: image,
                transforms.Resize((self.cfg.task.patch_image_size, self.cfg.task.patch_image_size), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
            ]
        )

        patch_image = patch_resize_transform(image).unsqueeze(0)
        patch_mask = torch.tensor([True])

        bos_item, eos_item, pad_idx = self.__text_preprocess()
        src_text = self.__encode_text(" what does the image describe?", bos_item, eos_item, append_bos=True, append_eos=True).unsqueeze(0)
        src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
        
        sample_image = {
            "id":np.array(['42']),
            "net_input": {
                "src_tokens": src_text,
                "src_lengths": src_length,
                "patch_images": patch_image,
                "patch_masks": patch_mask
            }
        }
        return sample_image
    
    def inference_image(self, image: Image.Image):
        sample = self.__construct_sample_image(image)
        sample = utils.move_to_cuda(sample) if self.__use_cuda else sample
        sample = utils.apply_to_sample(self.__apply_half, sample) if self.__fp16 else sample

        with torch.no_grad():
            result, scores = eval_caption(self.task, self.generator, self.models, sample)

        caption = result[0]['caption']

        return caption

    @staticmethod
    def load_images_from_directory( image_dir: str) -> Image.Image:
        image_files = os.listdir(image)
        for img_file in image_files:
            image_path = os.path.join(image_dir, img_file)
            image = Image.open(image_path)

            yield image

    @staticmethod
    def load_image(image_path: str):
        image = Image.open(image_path)
        image = image.convert("RGB")
        return image
    
    @staticmethod
    def save_captioned_image(image: Image.Image, caption: str, save_dir: str):

        captioned_text = f"Caption: {caption}"
        new_image = Image.new('RGB', (image.size[0], image.size[1] + 55), color='black')
        new_image.paste(image, (0, 0))

        draw = ImageDraw.Draw(new_image)
        font = ImageFont.truetype('./fonts/GenJyuuGothic-Light.ttf', size=23)
        cpation_pos = (8, image.size[1] + 13)

        draw.text(cpation_pos, captioned_text, font=font, fill=(0, 250, 0, 0))

        # new_image.show()
        new_image.save(save_dir)

def getConfigurations():
        parser = argparse.ArgumentParser()
        parser.add_argument('--sample_image_path', default='./sample_images', help="Sample Image Path for inferencing...")
        parser.add_argument('--save_dir', default='.ImageResult_withCaptions', help="Path to save captioned result images")

        args = parser.parse_args()

        with open('./configs/ofa_inference.yaml') as cfg:
            config: Dict = yaml.load(cfg, yaml.SafeLoader)

        return args, config
    

def main():

    args, config = getConfigurations()
    caption_inferencer = ImgCaptionInferencer(config)

    # image = caption_inferencer.load_image(args.sample_image_path)

    # start_time = time.time()

    # caption = caption_inferencer.inference_image(image)

    # end_time = time.time()

    # fps = 1/(end_time-start_time)

    # print("[FPS]: {}".format(fps))

    # filen = os.path.basename(args.sample_image_path)
    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir, exist_ok=True)
    # output_dir = os.path.join(args.save_dir, filen)

    # caption_inferencer.save_captioned_image(image, caption, output_dir)
    

    start_time = time.time()
    for image_pth in glob.glob(args.sample_image_path + '/*'):

        image = caption_inferencer.load_image(image_pth)
        caption = caption_inferencer.inference_image(image)
        print("Caption: ", caption)
        end_time = time.time()

        if end_time > start_time:
            fps = 1/(end_time - start_time)
            print("[FPS]: {}".format(fps))
        
        start_time = end_time

       
if __name__ == "__main__":
    main()