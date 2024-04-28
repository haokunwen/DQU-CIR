# DQU-CIR
### [SIGIR 2024] - Simple but Effective Raw-Data Level Multimodal Fusion for Composed Image Retrieval

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2404.15875)
![GitHub Repo stars](https://img.shields.io/github/stars/haokunwen/DQU-CIR?style=social)

This is the official implementation of our paper "Simple but Effective Raw-Data Level Multimodal Fusion for Composed Image Retrieval".

### Requirements

* python == 3.9.18
* pytorch == 2.0.1
* open-clip-torch == 2.23.0
* opencv-python == 4.9.0
* CUDA == 12.4
* 1 A100-40G GPU

### Datasets

As our method focuses on the raw-data level multimodal fusion, the dataset preparation steps might require some extra attention. We've released all the necessary files for you to download easily!

- FashionIQ:
  - Obtain the FashionIQ dataset by following the instructions in the [official repository](https://github.com/XiaoxiaoGuo/fashion-iq) or by downloading a version hosted on [Google Drive](https://drive.google.com/drive/folders/14JG_w0V58iex62bVUHSBDYGBUECbDdx9?usp=sharing), as provided by the authors of [CosMo](https://github.com/postBG/CosMo.pytorch). Ensure the dataset is stored in the folder `./data/FashionIQ`.
  - Once the dataset is downloaded, preprocess the images by running the [`resize_images.py`](https://github.com/XiaoxiaoGuo/fashion-iq/blob/master/start_kit/resize_images.py) script. The resized images should be saved in the `./data/FashionIQ/resized_image` directory.
  - We found that there are many typos in the modification text, so we use the [pyspellchecker](https://pypi.org/project/pyspellchecker/) tool to automatically correct them. The corrected files are available at `./data/FashionIQ/captions/correction_dict_{dress/shirt/toptee}.json`.
  - **For obtaining the Unified Textual Query:** We generate the image captions by [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b). The generated image caption files are located at `./data/FashionIQ/captions/image_captions_{dress/shirt/toptee}_{train/val}.json`.
  - **For obtaining the Unified Visual Query:** We extract the key words description of the target image from the modification text by Google DeepMind's [Gemini API](https://deepmind.google/technologies/gemini/#build-with-gemini) (gemini-pro-v1), with the prompt like: 
    ```
    Analyze the provided text, which details the differences between two images. Extract and list the distinct features of the target image mentioned in the caption, separating each feature with a comma. Ensure to eliminate any redundancies and correct typos. For instance, given the input 'has short sleeves and is red and has belt and is red', the response should be 'short sleeves, red, belt'. If the input is 'change from red to blue', respond with 'blue'. If there are no changes, respond with 'same'. The input text for this task is:
    ```
    The extracted key words files are available at `./data/FashionIQ/captions/keywords_in_mods_{dress/shirt/toptee}.json`.

- Shoes:
  - We noticed that the images of the Shoes dataset are currently unavailable from the [original source](https://github.com/XiaoxiaoGuo/fashion-retrieval/tree/master/dataset), so we provided it on [Google Drive](https://drive.google.com/file/d/18DEWXvuyp2vXHv4tAw6fcD2ehEtrvyIL/view?usp=sharing). Unzip the file and make the images by their names inside the `./data/Shoes/womens_*` folders.
  - We also corrected the modification text in Shoes by the [pyspellchecker](https://pypi.org/project/pyspellchecker/) tool. The file is available at `./data/Shoes/correction_dict_shoes.json`.
  - **For obtaining the Unified Textual Query:** Although Shoes dataset already contains the image captions at `./data/Shoes/captions_shoes.json`, we also generate the image captions by [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b). The generated image caption file is located at `./data/Shoes/image_captions_shoes.json`.
  - **For obtaining the Unified Visual Query:** Similar to FashionIQ, We extract the key words descriptions with the prompt like: 
    ```
    Analyze the provided text, which details the differences between two images. Extract and list the distinct features of the target image mentioned in the caption, separating each feature with a comma. Ensure to eliminate any redundancies and correct typos. For instance, given the input 'has long heels and is black instead of white', the response should be 'long heels, black'. If the input is 'change from red to blue', respond with 'blue'. If there are no changes, respond with 'same'. The input text for this task is: 
    ```
    The extracted key words files are available at `./data/Shoes/keywords_in_mods_shoes.json`.

- CIRR:
  - Download the CIRR dataset following the instructions in the [official repository](https://github.com/Cuberick-Orion/CIRR). Ensure the unzipped files are stored in the folder `./data/CIRR`.
  - **For obtaining the Unified Textual Query:** Similarly, we generate the image captions by [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b). The generated image caption files are located at `./data/CIRR/image_captions_cirr_{train/val/test1}.json`.
  - **For obtaining the Unified Visual Query:** Similarly, we extract the key words description of the target image with the prompt like: 
    ```
    Based solely on the input text which is to describe the difference between two images, extract the key features from the caption related to the target image. Provide the features separated by a comma. Avoid redundancy and revise the typos. For example, if the input text is 'instead of an old fortress with a rampart, an Orthodox church with a courtyard', you should response 'orthodox church, courtyard'. If the input text is 'One dog sitting on a bed looking at the camera', you should response 'one dog, sit on a bed, look at camera'. The input text is: 
    ```
    The extracted key words files are available at `./data/CIRR/keywords_in_mods_cirr_{train/val/test1}.json`.

- Fashion200K:
  - Download the Fashion200K dataset following the instructions in the [official repository](https://github.com/xthan/fashion-200k). Ensure the unzipped files are stored in the folder `./data/Fashion200K`.
  - Note that Fashion200K dataset conveniently provides reference image captions. Additionally, the modification text follows a template-based approach (e.g., "replace A with B"). This eliminates the need for BLIP-2 to generate image descriptions and Gemini to extract keywords.

### Usage

- FashionIQ: ```python train.py --dataset {'dress'/'shirt'/'toptee'} --lr 1e-4 --clip_lr 1e-6 --fashioniq_split={'val-split'/'original-split'}```  
- Shoes: ```python train.py --dataset 'shoes' --lr 5e-5 --clip_lr 5e-6```
- CIRR:  
  Training: ```python train.py --dataset 'cirr' --lr 1e-4 --clip_lr 1e-6 ```  
  Testing: ```python cirr_test_submission.py --i xx ```  
- Fashion200K: ```python train.py --dataset 'fashion200k' --lr 1e-4 --clip_lr 1e-6```

### Citation
If you find this work useful in your research, please consider citing:
```bibtex
@inproceedings{dqu_cir,
    author = {Wen, Haokun and Song, Xuemeng and Chen, Xiaolin and Wei, Yinwei and Nie, Liqiang and Chua, Tat-Seng},
    title = {Simple but Effective Raw-Data Level Multimodal Fusion for Composed Image Retrieval},
    booktitle = {Proceedings of the International ACM SIGIR Conference on Research and Development in Information Retrieval},
    pages = {},
    publisher = {{ACM}},
    year = {2024}
}
```
