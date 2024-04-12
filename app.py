
import gradio as gr
import numpy as np 
import config
from PIL import Image
from utils import *
import torch.optim as optim 
import matplotlib.pyplot as plt
from model import generator, Discriminator
from torchvision.utils import save_image
gen_A = generator(img_channels=3).to(config.DEVICE)
gen_B = generator(img_channels=3).to(config.DEVICE)
opt_gen = optim.Adam(
        list(gen_A.parameters()) + list(gen_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )
load_checkpoint(config.CHECKPOINT_GEN_B, gen_B, opt_gen, config.LEARNING_RATE),
load_checkpoint(config.CHECKPOINT_GEN_A, gen_A, opt_gen, config.LEARNING_RATE),
def winter2summer(image:np.ndarray):
    # pil_image = np.array(Image.fromarray(image).convert("RGB"))
    if config.transforms_app:
        augmentations = config.transforms_app(image=image)
        winter_image = augmentations['image'].to(config.DEVICE)
    summer_image = gen_B(winter_image)
    summer_image = summer_image*0.5 + 0.5
    summer_image = summer_image.cpu().detach().numpy()
    summer_image = np.transpose(summer_image,(1, 2, 0))
    return summer_image
def summer2winter(image:np.ndarray):
    if config.transforms_app:
        augmentations = config.transforms_app(image=image)
        summer_image = augmentations['image'].to(config.DEVICE)
    winter_image = gen_A(summer_image)
    winter_image = winter_image*0.5 + 0.5
    winter_image = winter_image.cpu().detach().numpy()
    winter_image = np.transpose(winter_image,(1, 2, 0))
    return winter_image

interface_css = """
footer {visibility: hidden}
"""
def main():
    with gr.Blocks(css=interface_css,theme=gr.themes.Soft()) as demo:
        gr.Markdown("""<center><font size=10>Cycle GAN demo by masquerlin</center>""")
        gr.Markdown("""<center><font size=8>winter to summer</center>""")
        with gr.Row(equal_height=False):
            
            image_origin_winter = gr.Image(type='numpy', height=512)
            sumbit_winter2summer = gr.Button("🚀 winter to summer")
            img_output_summer = gr.Image(type='numpy', height=512)
        gr.Markdown("""<center><font size=8>summer to winter</center>""")    
        with gr.Row(equal_height=False):
            
            image_origin_summer = gr.Image(type='numpy', height=512)
            with gr.Column():

                sumbit_summer2winter = gr.Button("🚀 summer to winter")

            img_output_winter = gr.Image(type='numpy', height=512)
            

        sumbit_winter2summer.click(winter2summer, image_origin_winter, img_output_summer)
        sumbit_summer2winter.click(summer2winter, image_origin_summer, img_output_winter)

    demo.queue(api_open=False).launch(max_threads=10, height=800, share=False)
if __name__ == "__main__":
    main()
