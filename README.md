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

### Preparation
- Please go to OFA folder
  ```
  cd OFA
  ```
- Please Add Submodule by running the following command
  ```
  git submodule init
  ```
- Pull latest commits from sub modules' branch
  ```
  git submodule update --init --remote --merge --recursive
  ```
  - Please Download Weight file
  ```
  wget https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/caption_base_best.pt
  mkdir weight_checkpoints
  cp caption_base_best.pt weight_checkpoints
  ```

### Environment
- Please use one of the following approach to make environments.
  #### With Conda
    ```
    conda create -n ImageCaptioningEnv python=3.7.4
    conda activate ImageCaptioningEnv
  
    pip install -r requirements.txt
    ```
  
  #### With Docker
    ```
    make docker-build
    
    make docker-run
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
