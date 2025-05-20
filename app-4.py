import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import lru_cache

@lru_cache(maxsize=1)
def get_model():
    return tf.keras.models.load_model("trained_car_model_midterm.h5")


#model = load_model("trained_car_model_midterm.h5")


class_labels = {
0: 'Acura_Acura ILX_2013',
1: 'Acura_Acura MDX_2014',
2: 'Acura_Acura NSX_2012',
3: 'Acura_Acura RDX_2013',
4: 'Acura_Acura RLX_2013',
5: 'Acura_Acura TLX_2015',
6: 'Acura_Acura TL_2012',
7: 'Acura_Acura ZDX_2009',
8: 'Alfa Romeo_8C_2009',
9: 'Alfa Romeo_Giulietta_2014',
10: 'Alfa Romeo_MiTo_2014',
11: 'Aston Martin_DB9_2007',
12: 'Aston Martin_DBS_2009',
13: 'Aston Martin_ONE-77_2010',
14: 'Aston Martin_Rapide_2013',
15: 'Aston Martin_V12 Vantage_2014',
16: 'Aston Martin_Vanquish_2013',
17: 'Aston Martin_Virage_2012',
18: 'Audi_Audi A3 sedan_2014',
19: 'Audi_Audi A5 convertible_2010',
20: 'Audi_Audi A5 coupe_2014',
21: 'Audi_Audi A5 hatchback_2010',
22: 'Audi_Audi A5 hatchback_2014',
23: 'Audi_Audi Q3_2015',
24: 'Audi_Audi Q5_2010',
25: 'Audi_Audi RS Q3_2014',
26: 'Audi_Audi RS4_2013',
27: 'Audi_Audi RS5_2013',
28: 'Audi_Audi RS7_2014',
29: 'Audi_Audi S1_2014',
30: 'Audi_Audi S3 sedan_2015',
31: 'Audi_Audi S4_2013',
32: 'Audi_Audi S7_2013',
33: 'Audi_Audi S8_2014',
34: 'Audi_Audi SQ5_2013',
35: 'Audi_Audi e-tron_2010',
36: 'Audi_Audi quattro_2010',
37: 'Audi_Crosslane Coupe_2012',
38: 'BAW_BAW BJ40_2014',
39: 'BAW_BAW E Series hatchback_2012',
40: 'BAW_BAW E Series sedan _2013',
41: 'Baihc_Yusheng 007_2011',
42: 'Baojun_Baojun 610_2014',
43: 'Baojun_Baojun 730_2014',
44: 'Beiqi New Energy_BAW E150 EV_2014',
45: 'Beiqihuansu_Huansu S2_2014',
46: 'Beiqihuansu_Huansu S3_2014',
47: 'Beiqiweiwang_Weiwang 205_2013',
48: 'Beiqiweiwang_Weiwang 306_2011',
49: 'Beiqiweiwang_Weiwang 307_2014',
50: 'Beiqiweiwang_Weiwang M20_2014',
51: 'Bentley_Brooklands_2008',
52: 'Bentley_Flying Spur_2013',
53: 'Bentley_Mulsanne_2011',
54: 'Benz_AMG GT_2015',
55: 'Benz_Benz A Class AMG_2014',
56: 'Benz_Benz A Class_2013',
57: 'Benz_Benz B Class_2012',
58: 'Benz_Benz C Class AMG_2015',
59: 'Benz_Benz CL Class AMG_2011',
60: 'Benz_Benz CLA Class AMG_2014',
61: 'Benz_Benz CLA Class_2014',
62: 'Benz_Benz CLS Class AMG_2015',
63: 'Benz_Benz CLS Class_2013',
64: 'Benz_Benz E Class AMG_2014',
65: 'Benz_Benz E Class convertible_2010',
66: 'Benz_Benz E Class couple_2014',
67: 'Benz_Benz E Class_2014',
68: 'Benz_Benz F800_2010',
69: 'Benz_Benz G Class AMG_2013',
70: 'Benz_Benz G Class_2013',
71: 'Benz_Benz GL Class AMG_2013',
72: 'Benz_Benz GL Class_2013',
73: 'Benz_Benz GLA Class AMG_2015',
74: 'Benz_Benz GLA Class_2015',
75: 'Benz_Benz M Class_2014',
76: 'Benz_Benz R Class_2014',
77: 'Benz_Benz S Class AMG_2015',
78: 'Benz_Benz SL Class AMG_2013',
79: 'Benz_Benz SLK Class_2011',
80: 'Benz_Benz V Class_2015',
81: 'Benz_Coupe SUV_2014',
82: 'Benz_Sprinter_2012',
}

def classify_image_(image):
    try:
        model = get_model()
        image = image.convert("RGB")
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction[0])
        predicted_label = class_labels.get(predicted_class, "Unknown")

        return img_resized, f"The car brand in the image is: {predicted_label}"
    except Exception as e:
        return None, f"Error: {str(e)}"

title = "Car Model Classification via CNN"
description = "Upload an image of a car to classify its brand, model, and year."


def get_examples(base_path="test_images_compcars"):
    examples = []
    for class_folder in os.listdir(base_path):
        class_folder_path = os.path.join(base_path, class_folder)
        if os.path.isdir(class_folder_path):
            image_files = [f for f in os.listdir(class_folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                example_path = os.path.join(base_path, class_folder, image_files[0])
                examples.append([example_path])
    return examples
    
examples = get_examples()


demo = gr.Interface(
    fn=classify_image_,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(label="Processed Image"), gr.Textbox(label="Predicted Class")],
    title=title,
    description=description,
    examples=examples
)

if __name__ == "__main__":
    demo.launch(share = True)