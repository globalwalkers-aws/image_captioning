import argparse, yaml, os, utils
from pathlib import Path
from models.tokenization_bert import BertTokenizer
from models.model_caption_mplug import MPLUG
import torch
import torch.backends.cudnn as cudnn
from optim import create_optimizer, create_two_optimizer

class ImageCaptionModel:

    def __init__( self, args, config):
        utils.init_distributed_mode( args )
        self.deivce     = torch.device( args.device )
        cudnn.benchmark = True
        self.tokenizer  = BertTokenizer.from_pretrained(args.text_encoder)
        self.model      = MPLUG( config = config, tokenizer=self.tokenizer )
        self.model      = self.model.to(self.deivce)
        self.optimiser  = create_optimizer( utils.AttrDict(config['optimizer']), self.model )
        self.checkpoint = torch.load( args.checkpoint, map_location="cpu" )
        try:
            self.state_dict = self.checkpoint['model']
        except:
            self.state_dict = self.checkpoint['module']

        print( self.state_dict )
        print( self.checkpoint )

def getConfigurations():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_mplug_base.yaml')
    parser.add_argument('--checkpoint', default='')
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

def main():

    args, config = getConfigurations()
    image_caption_model = ImageCaptionModel( args, config )

if __name__ == "__main__":
    main()