import argparse, yaml, os, utils, torch, glob
from pathlib import Path
from models.tokenization_bert import BertTokenizer
from models.model_caption_mplug import MPLUG
import torch.nn as nn
import torch.backends.cudnn as cudnn
from optim import create_optimizer, create_two_optimizer
from models.vit import resize_pos_embed
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

class ImageCaptionModel:

    def __init__( self, args, config):
        utils.init_distributed_mode( args )
        self.device     = torch.device( args.device )
        cudnn.benchmark = True
        self.tokenizer  = BertTokenizer.from_pretrained(args.text_encoder)
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
        pos_embed = resize_pos_embed(self.state_dict['visual_encoder.visual.positional_embedding'].unsqueeze(0),
                                                   pos_embed.unsqueeze(0))
        self.state_dict['visual_encoder.visual.positional_embedding'] = pos_embed
        self.model.load_state_dict( self.state_dict, strict=False )
        self.model.eval()
        self.model.to( self.device )

        print("Model loaded")

    def inference( self, image_data ):
        top_ids, top_probs = self.model( image_data, "caption this photo", train=False )
        
        print( top_ids )
        for id in top_ids:
            print(id[0])
            ans = self.tokenizer.decode(id[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
            print( ans )
def getConfigurations():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_mplug_base.yaml')
    parser.add_argument('--checkpoint', default='./mplug_base.pth')
    parser.add_argument('--output_dir', default='output/mplug_base')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--min_length', default=1, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--max_length', default=10, type=int)
    parser.add_argument('--max_input_length', default=25, type=int)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--do_two_optim', action='store_true')
    parser.add_argument('--add_object', action='store_true')
    parser.add_argument('--do_amp', action='store_true')
    parser.add_argument('--no_init_decocde', action='store_true')
    parser.add_argument('--do_accum', action='store_true')
    parser.add_argument('--accum_steps', default=4, type=int)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    config["min_length"] = args.min_length
    config["max_length"] = args.max_length
    config["add_object"] = args.add_object
    config["beam_size"] = args.beam_size
    config['optimizer']['lr'] = args.lr
    config['schedular']['lr'] = args.lr
    config['text_encoder'] = args.text_encoder
    config['text_decoder'] = args.text_decoder
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    return args, config

def load_image(image, image_size):
    device    = "cuda:0"
    raw_image = Image.open(str(image)).convert('RGB')

    w, h = raw_image.size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image

def main():

    args, config        = getConfigurations()
    image_caption_model = ImageCaptionModel( args, config )

    image_folder        = "./sample_images/"
    for image in glob.glob( image_folder + '/*' ):
        image_data = load_image( image, image_size=384 )
        image_caption_model.inference( image_data )

if __name__ == "__main__":
    main()