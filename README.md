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

- Download pretained model and place it inside the BLIP folder

```
https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth
```

- Create BLIP docker environment

```
./run_system.sh
```

- Put some sample images inside sample_images/ folder

```
python3 inference_blip.py
```
