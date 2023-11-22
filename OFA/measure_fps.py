from inference import ImgCaptionInferencer
import argparse, yaml, time, glob
from typing import Dict

def getArgs():
        parser = argparse.ArgumentParser()
        parser.add_argument('--sample_image_path', default='./sample_images', help="Sample Image Path for inferencing...")

        args = parser.parse_args()

        with open('./configs/ofa_inference.yaml') as cfg:
            config: Dict = yaml.load(cfg, yaml.SafeLoader)

        return args, config

def measureFPS():
      args, config = getArgs()
      captionInferencer = ImgCaptionInferencer(config)

      start_time = time.time()

      for image_pth in glob.glob(args.sample_image_path + '/*'):
            image = captionInferencer.load_image(image_pth)
            caption = captionInferencer.inference_image(image)
            print("Caption: ", caption)
            end_time = time.time()

            if end_time > start_time:
                fps = 1/(end_time - start_time)
                print("[FPS]: {}".format(fps))
            
            start_time = end_time
      
       
if __name__ == "__main__":
    measureFPS()