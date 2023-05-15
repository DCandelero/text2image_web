import streamlit as st
from main import load_model, generate_images
import re

st.set_page_config(page_title='text2image', page_icon="figures/framed_picture_icon.png", layout="centered")

with st.sidebar:
    models = ["[birds] scratch (1 epoch)", "[birds] scratch finetune (1 epoch)",
              "[birds] pretrained (590 epochs)", "[birds] pretrained finetune (550 epochs)", 
            #   "[coco] pretrained finetune (590 epochs)", "[coco] pretrained (550 epochs)", 
            #   "[coco] scratch (1 epochs)", "[coco] scratch finetune (1 epochs)"
            ]
    models_path = ["models/birds/netG_001.pth", "models/birds/netG_591.pth", 
              "models/birds/netG_590.pth", "models/birds/netG_550.pth",
            #   "[coco] pretrained finetune (590 epochs)", "[coco] pretrained (550 epochs)", 
            #   "[coco] scratch (1 epochs)", "[coco] scratch finetune (1 epochs)"
            ]
    model = st.selectbox(
        label="Choose dataset to be used for image generation",
        options=models,
        index=0
    )
    model_idx = models.index(model)
    model_path = models_path[model_idx]


    if model.startswith("[birds]"):
        model_help_text = '''
            This model is used to generate images 
            of birds. Please enter a sentence to 
            generate images related to this theme, 
            for example:
            "This bird is black with a white chest 
            and belly and has a long neck"
        '''
    else:
        model_help_text = ""
    st.code(model_help_text, language="markdown")

    st.divider()

    n_generated_images = st.number_input(
        label='Choose the number of images to be generated',
        min_value=1,
        max_value=6,
    )

@st.cache_resource
def loading_model(model_path:str):
    return load_model(model_path)
text_encoder, netG, device, wordtoix = loading_model(model_path)

# Body
st.title('Text to image generator')

text_caption = st.text_input('Type a sentence to generate an image',
                     help=model_help_text)
text_caption = re.sub(r'[^a-zA-Z0-9\s]+', '', text_caption)

if text_caption:
    generated_images = generate_images(text_encoder, netG, device, wordtoix, [text_caption], n_generated_images)
    
    n_cols = 3
    col1, col2, col3 = st.columns(n_cols)
    for i in range(n_generated_images):
        if i % n_cols == 0:
            col1.image(generated_images[i])
        elif i % n_cols == 1: 
            col2.image(generated_images[i])
        elif i % n_cols == 2: 
            col3.image(generated_images[i])


