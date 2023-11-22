### Researching Image captioning

This repository contains the implementation for three image captioning models ( mPLUG, BLIP and OFA )


## mPLUG 

- Please go inside mPLUG folder

```
cd mPLUG
```

- Create mPLUG docker environment

```
./run_system.sh
```

- Put some sample images inside sample_images/ folder

```
python3 inference.py
```

## BLIP

- Please go inside BLIP folder

```
cd BLIP
```

- Create BLIP docker environment

```
./run_system.sh
```

- Put some sample images inside sample_images/ folder

```
python3 inference_blip.py
```

## OFA

### Environment
- Please use one of the following command to make environments.

  #### With Docker
    ```
    make docker-build
    ```
### Run Inference

- Please run the following command to download weigh file, and run docker environment to start inferencing on sample images
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
