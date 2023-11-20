"""
Download the weights in ./checkpoints beforehand for fast inference
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_vqa.pth
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth
"""

from pathlib import Path

from PIL import Image
import torch, glob, numpy, cv2, time
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder
from models.blip_vqa import blip_vqa
from models.blip_itm import blip_itm

BLACK_BACKGROUND_HEIGHT = 50
OUTPUT_WIDTH            = 720
OUTPUT_HEIGHT           = 480

class Predictor:
    def __init__(self):
        self.device = "cuda:0"

        self.models = {
            'image_captioning': blip_decoder(pretrained='model_base_caption_capfilt_large.pth',
                                             image_size=384, vit='base'),
        }

    def predict(self, image, task, question, caption):
        if task == 'visual_question_answering':
            assert question is not None, 'Please type a question for visual question answering task.'
        if task == 'image_text_matching':
            assert caption is not None, 'Please type a caption for mage text matching task.'

        im, cv_img = load_image(image, image_size=480 if task == 'visual_question_answering' else 384, device=self.device)
        model = self.models[task]
        model.eval()
        model = model.to(self.device)
        start_time = time.time()

        if task == 'image_captioning':
            with torch.no_grad():
                caption = model.generate(im, sample=False, num_beams=3, max_length=20, min_length=5)
                end_time = time.time()
                fps      = 1 / ( end_time - start_time )
                print( f"FPS : {fps}")
                return caption[0], cv_img

        # image_text_matching
        itm_output = model(im, caption, match_head='itm')
        itm_score = torch.nn.functional.softmax(itm_output, dim=1)[:, 1]
        itc_score = model(im, caption, match_head='itc')
        return f'The image and text is matched with a probability of {itm_score.item():.4f}.\n' \
               f'The image feature and text feature has a cosine similarity of {itc_score.item():.4f}.'


def load_image(image, image_size, device):
    raw_image = Image.open(str(image)).convert('RGB')
    cv2_image = numpy.array( raw_image )
    cv2_image = cv2_image[:,:,::-1].copy()
    cv2_image  = cv2.resize( cv2_image, ( OUTPUT_WIDTH, OUTPUT_HEIGHT ))

    w, h = raw_image.size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image, cv2_image

def generateDisplayImage( generated_caption, cv2_image ):
    display_text     = f"Caption: {generated_caption}"
    black_background = numpy.zeros([ BLACK_BACKGROUND_HEIGHT, cv2_image.shape[1], 3], dtype=numpy.uint8)
    cv2.putText( black_background, display_text, (int(BLACK_BACKGROUND_HEIGHT/2), int(BLACK_BACKGROUND_HEIGHT/2)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 250, 0), 1, cv2.LINE_AA )
    stack_image = cv2.vconcat( [black_background, cv2_image] )
    return stack_image

def main():

    sample_dir = './sample_images/'
    blip_model = Predictor()
    for image in glob.glob( sample_dir + '/*' ):
        caption, cv_img = blip_model.predict( image, "image_captioning", "what is in the image?", None )
        output_image    = generateDisplayImage( caption, cv_img )
        
if __name__ == "__main__":
    main()