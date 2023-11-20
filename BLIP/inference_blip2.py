import torch, glob
from PIL import Image
from lavis.models import load_model_and_preprocess

def load_image( image_path ):
    image = Image.open( image_path ).convert('RGB')
    return image

def load_model():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)
    return model, vis_processors, device

def generateCaption( model, vis_processors, image, device ):
    image = vis_processors["eval"](image).unsqueeze(0).to( device )
    ans = model.generate({"image": image, "prompt": "What is in the photo"})
    print( ans  )

def main():

    image_dir = "./sample_images/"
    model, processor, device = load_model()
    for image in glob.glob( image_dir + '/*' ):
        img_data = load_image( image )
        generateCaption( model, processor, img_data, device )

if __name__ == "__main__":
    main()
