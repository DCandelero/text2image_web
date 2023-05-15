# text2image_web

A web interface to generate images by captions. This application uses dl models implemented in [Text to Image Generation with Semantic-Spatial Aware GAN](https://arxiv.org/abs/2104.00567)

-----

## Main Requirements
* Python 3.10.0
All requirements can be found on requirements.txt, for easy usage we have a example below to set a virtual environment and run requirements for the implementation.
```
$ python -m venv venv
$ ./venv/Scripts/activate
$ pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
$ pip install -r requirements.txt
```
-----

## Download required content
Run download_content.py, with the following args below, to download birds encoders:
```
$ python download_content.py --dataset birds --content encoders
```

Run download_content.py, with the following args below, to download birds trained models:
```
$ python download_content.py --dataset birds --content models
```

## Run web application
```
$ streamlit run streamlit_app.py
```
