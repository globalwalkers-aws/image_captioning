# OFA
## Original Repository & Paper
* Repository -> [OFA Repository](https://github.com/OFA-Sys/OFA)
* Paper -> [Paper PDF](https://arxiv.org/pdf/2202.03052.pdf)

## Overview
* OFA is a Sequence-to-sequence learning framework that supports flexible modality tasks, and it achieved state-of-the-art performance on a variety of cross-modal and unimodal tasks.
* It not only supports cross-modal tasks like image generation, visual grounding and Image Captioning, but also supports image classification, language modeling tasks. 
* Because of its cross modalities, OFA can take both image and text as input, allowing it to generate more comprehensive and informative captions.
* In comparison with recent models that rely on extremely cross-modal datasets, OFA is pretrained on the publicly available small scale datasets of 20M image-text pairs.
* OFA achieves new SOTAs in a series of cross-modal tasks including Image Captioning, and attain high competitive performance on uni-modal tasks.
* Apart from representing images, it also supports grounded question answering task like representing objects of bounding boxes localizing within images.
* It supports flexible size of model checkpoints which can help to choose between desired speed and accuracy of specific tasks.


### Environment
- Please use one of the following command to make environments.

  #### With Docker
    ```
    make docker-build
    ```
### Run Inference

- Please run the following commands for captioning inferencing
  #### Inference images on images under sample_images directory
     ```
     python inference.py --sample_image_path {path/to/sample.jpg} --save_dir {results/dir/}
     ```
     
  #### Measure FPS
    ```
    python measure_fps.py --sample_image_path {dir/to/sample_images}
    ```
