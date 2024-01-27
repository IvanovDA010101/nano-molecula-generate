import torch
import transformers
from diffusers import StableDiffusionPipeline

import gradio as gr
#import torch
from torch import autocast

from PIL import Image

pipe = StableDiffusionPipeline.from_pretrained("purpletyoma/sd-nano", safety_checker = None)
pipe = pipe.to("cpu")

description = \
"""
SD-NANO can create scanning electron microscopy (SEM) microphotographs of nanomaterials from parameters of any solution synthesis techniques for calcium carbonate system.
"""

def infer(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13): 
    prompt = [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13]
    text = "ca {}, co {}, hco {}, polytype {}, polymwt {}, polyperc {}, surftype {}, surfwt {}, solvtype {}, Solvvol {}, rpm {}, temp {}, time {}".format(*prompt)
    image = pipe(text).images[0]
    return image

print("Great sylvain ! Everything is working fine !")




gr.Interface(fn=infer, 
             inputs=[
                 gr.Textbox(label="Ca ion, mM", value = 33),
                 gr.Textbox(label="CO3 ion, mM", value = 110),
                 gr.Textbox(label="HCO3 ion, mM", value = 0),
                 gr.Textbox(label="Polymer type", value = 'PEG'),
                 gr.Textbox(label="Polymer Mwt, kDa", value = 8),
                 gr.Textbox(label="Polymer, % wt.", value = 1.1),
                 gr.Textbox(label="Surfactant type", value = 'Sodium dodecylsulfate'),
                 gr.Textbox(label="Surfactant, % wt.", value = 0.31),
                 gr.Textbox(label="Solvent type", value = None),
                 gr.Textbox(label="Solvent, % vol.", value = 0.0),
                 gr.Textbox(label="Stirring, rpm", value = 0),
                 gr.Textbox(label="Temperature, C", value = 65),
                 gr.Textbox(label="Synthesis time in seconds", value = 1959)],
             outputs=gr.Image(type='pil'),
             description=description,
             title="Nanomaterials Image Generation").launch()