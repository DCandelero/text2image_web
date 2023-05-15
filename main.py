# -*- encoding: utf-8 -*-
from __future__ import print_function
import multiprocessing

import os
import sys
import errno
import pickle

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from miscc.config import cfg, cfg_from_file
from sync_batchnorm import DataParallelWithCallback
from DAMSM import RNN_ENCODER, CNN_ENCODER
from model import NetG

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

multiprocessing.set_start_method('spawn', True)


def generate_images(text_encoder, netG, device, wordtoix, captions:list=[], number_of_images:int=1):
    """
    generate sample according to user defined captions.

    caption should be in the form of a list, and each element of the list is a description of the image in form of string.
    caption length should be no longer than 18 words.
    example captions see below
    """
    if len(captions) == 0:
        captions = ['this large bird is uniformly gray all over it also has a relatively large dark colored bill']

    # caption to idx
    # split string to word
    for c, i in enumerate(captions):
        captions[c] = i.split()

    caps = torch.zeros((len(captions), 18), dtype=torch.int64)

    for cl, line in enumerate(captions):
        for cw, word in enumerate(line):
            caps[cl][cw] = wordtoix[word.lower()]
    caps = caps.to(device)
    cap_len = []
    for i in captions:
        cap_len.append(len(i))

    caps_lens = torch.tensor(cap_len, dtype=torch.int64).to(device)

    model_dir = cfg.TRAIN.NET_G
    split_dir = 'user_caption_generated'
    netG.load_state_dict(torch.load(model_dir))
    netG.eval()

    batch_size = len(captions)
    generated_images = []
    for step in range(number_of_images):

        hidden = text_encoder.init_hidden(batch_size)
        words_embs, sent_emb = text_encoder(caps, caps_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

        #######################################################
        # (2) Generate fake images
        ######################################################
        with torch.no_grad():
            # noise = torch.randn(1, 100) # using fixed noise
            # noise = noise.repeat(batch_size, 1)
            # use different noise
            noise = []
            for i in range(batch_size):
                noise.append(torch.randn(1, 100))
            noise = torch.cat(noise,0)
            
            noise = noise.to(device)
            fake_imgs, stage_masks = netG(noise, sent_emb)
            stage_mask = stage_masks[-1]
        for j in range(batch_size):
            im = fake_imgs[j].data.cpu().numpy()
            # [-1, 1] --> [0, 255]
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)


            generated_images.append(im)

    return generated_images

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def load_model(model_path:str='models/birds/netG_001.pth'):
    cfg_from_file('cfg/bird.yml')

    if model_path != '':
        cfg.TRAIN.NET_G = model_path

    with open('metadata/birds/wordtoix.pkl', 'rb') as fp:
        wordtoix = pickle.load(fp)

    # Set device and network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = NetG(cfg.TRAIN.NF, 100).to(device)
    netG = DataParallelWithCallback(netG)

    # Load text encoder
    text_encoder = RNN_ENCODER(len(wordtoix), nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.cuda()
    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()

    # Load image encoder
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    img_encoder_path = cfg.TEXT.DAMSM_NAME.replace('text_encoder', 'image_encoder')
    state_dict = \
        torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
    image_encoder.load_state_dict(state_dict)
    image_encoder.cuda()
    for p in image_encoder.parameters():
        p.requires_grad = False
    print('Load image encoder from:', img_encoder_path)
    image_encoder.eval()

    return text_encoder, netG, device, wordtoix