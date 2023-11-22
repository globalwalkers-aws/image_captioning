# OFA
## Original Repository & Paper
* Repository -> [OFA Repository](https://github.com/OFA-Sys/OFA)
* Paper -> [Paper PDF](https://arxiv.org/pdf/2202.03052.pdf)

## Inferencing OFA
### Build Docker Environment
- Please run the following command to build Docker environment.
    ```
    make docker-build
    ```
### Run Inference

- Please run the following command to download weigh file, and run docker environment to start inferencing on sample images.
  ```
  ./run_system.sh
  ```
  #### Inference images on images under sample_images directory
     ```
     python3 inference.py --sample_image_path {path/to/sample.jpg} --save_dir {results/dir/}
     ```
     
  #### Measure FPS
    ```
    python3 measure_fps.py --sample_image_path {dir/to/sample_images}
    ```
