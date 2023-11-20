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

### Preparation
#### With Conda
```
conda create -n ImageCaptioningEnv python=3.7.4
conda activate ImageCaptioningEnv

pip install -r requirements.txt
```

#### With Docker
```

```

#### Run inference

