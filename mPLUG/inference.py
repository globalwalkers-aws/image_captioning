import argparse, yaml, os, utils, torch, glob, cv2, numpy, time
from pathlib import Path
from models.tokenization_bert import BertTokenizer
from models.model_caption_mplug import MPLUG
import torch.nn as nn
import torch.backends.cudnn as cudnn
from optim import create_optimizer
from models.vit import resize_pos_embed
from PIL import Image
from torchvision import transforms

BLACK_BACKGROUND_HEIGHT = 50
OUTPUT_WIDTH            = 720
OUTPUT_HEIGHT           = 480

class ImageCaptionModel:

    def __init__( self, args, config):
        print(f"Loading mPLUG model . . .")
        utils.init_distributed_mode( args )
        self.device     = torch.device( args.device )
        cudnn.benchmark = True
        self.tokenizer  = BertTokenizer.from_pretrained( config['text_encoder'])
        self.model      = MPLUG( config = config, tokenizer=self.tokenizer )
        self.model      = self.model.to(self.device)
        self.optimiser  = create_optimizer( utils.AttrDict(config['optimizer']), self.model )
        self.checkpoint = torch.load( args.checkpoint, map_location="cpu" )
        
        try:
            self.state_dict = self.checkpoint['model']
        except:
            self.state_dict = self.checkpoint['module']
            
        num_patches = int(config["image_res"] * config["image_res"]/(16*16))
        pos_embed   = nn.Parameter(torch.zeros(num_patches + 1, 768).float())
        pos_embed   = resize_pos_embed(self.state_dict['visual_encoder.visual.positional_embedding'].unsqueeze(0),
                                                   pos_embed.unsqueeze(0))
        self.state_dict['visual_encoder.visual.positional_embedding'] = pos_embed
        self.model.load_state_dict( self.state_dict, strict=False )
        self.model.eval()
        self.model.to( self.device )

        print(f"Model loaded: {args.checkpoint}")

    def generateDisplayImage( self, generated_caption, cv2_image ):
        display_text     = f"Caption: {generated_caption}"
        black_background = numpy.zeros([ BLACK_BACKGROUND_HEIGHT, cv2_image.shape[1], 3], dtype=numpy.uint8)
        cv2.putText( black_background, display_text, (int(BLACK_BACKGROUND_HEIGHT/2), int(BLACK_BACKGROUND_HEIGHT/2)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 250, 0), 1, cv2.LINE_AA )
        stack_image = cv2.vconcat( [black_background, cv2_image] )
        return stack_image


    def inference( self, transfomred_image, cv2_image ):
        start_time = time.time()
        top_ids, _ = self.model( transfomred_image, "", train=False )
        cv2_image  = cv2.resize( cv2_image, ( OUTPUT_WIDTH, OUTPUT_HEIGHT ))
        for id in top_ids:
            ans              = self.tokenizer.decode(id[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
            end_time         = time.time()
            fps              = 1 / ( end_time - start_time )
            display_image    = self.generateDisplayImage( ans, cv2_image )
            cv2.imshow('output', display_image)
            cv2.waitKey(0)

    @staticmethod
    def load_image(image, image_size):
        device    = "cuda:0"
        raw_image = Image.open(str(image)).convert('RGB')
        cv2_image = numpy.array( raw_image )
        cv2_image = cv2_image[:,:,::-1].copy()

        w, h = raw_image.size

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size) ),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        image = transform(raw_image).unsqueeze(0).to(device)
        return image, cv2_image
        

def getConfigurations():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_mplug_base.yaml')
    parser.add_argument('--checkpoint', default='./mplug_base.pth')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--min_length', default=10, type=int)
    parser.add_argument('--max_length', default=25, type=int)
    parser.add_argument('--max_input_length', default=25, type=int)

    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # assign the config variables needed for model initialisation
    config["min_length"] = args.min_length
    config["max_length"] = args.max_length
    config['text_encoder'] = "bert-base-uncased"
    config['text_decoder'] = "bert-base-uncased"
    config['beam_size']    = 5
    config['optimizer']['lr'] = 2e-5

    return args, config

def main():

    args, config        = getConfigurations()
    image_caption_model = ImageCaptionModel( args, config )
    image_folder        = "./sample_images/"
    for image in glob.glob( image_folder + '/*' ):
        transformed_image, cv2_image = image_caption_model.load_image( image, image_size=config['image_res'] )
        image_caption_model.inference( transformed_image, cv2_image )

if __name__ == "__main__":
    main()