import streamlit as st
from PIL import Image, ExifTags
from io import BytesIO
from base64 import b64decode, b64encode
import requests
import random


CLIP_ENDPOINT = "https://cartoonizer-clip-test-4jkxk521l3v1.octoai.cloud/"
SD_ENDPOINT = "https://sd-demo-gcsv8y11zs17.octoai.cloud"

# PIL helper
def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

# PIL helper
def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def cartoonize_image(upload, model_name, strength, seed, loras, steps, extra_desc):
    input_img = Image.open(upload)
    try:
        # Rotate based on Exif Data
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif = input_img._getexif()
        if exif[orientation] == 3:
            input_img=input_img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            input_img=input_img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            input_img=input_img.rotate(90, expand=True)
    except:
        # Do nothing
        print("No rotation to perform based on Exif data")
    # Apply cropping and resizing to work on a square image
    cropped_img = crop_max_square(input_img)
    resized_img = cropped_img.resize((512, 512))
    col1.write("Original Image :camera:")
    col1.image(resized_img)

    # Prepare the JSON query to send to OctoAI's inference endpoint
    buffer = BytesIO()
    resized_img.save(buffer, format="png")
    image_out_bytes = buffer.getvalue()
    image_out_b64 = b64encode(image_out_bytes)

    # Prepare CLIP request
    clip_request = {
        "mode": "fast",
        "image": image_out_b64.decode("utf8"),
    }
    # Send to CLIP endpoint
    reply = requests.post(
        "{}/predict".format(CLIP_ENDPOINT),
        headers={"Content-Type": "application/json"},
        json=clip_request
    )
    # Retrieve prompt
    clip_reply = reply.json()["completion"]["labels"]

    # Uncomment if you want to edit the results of the CLIP model
    # Editable CLIP interrogator output
    # prompt = st.text_area("AI-generated, human editable label:", value=clip_reply)

    prompt = extra_desc + ", " + clip_reply
    # Prepare SD request for img2img
    sd_request = {
        "init_image": image_out_b64.decode("utf8"),
        "prompt": prompt, 
        "strength": float(strength)/10,
        # The rest below is hard coded
        "negative_prompt": "EasyNegative, (ugly:1.2), (worst quality, poor details:1.4), badhandv4, blurry",
        "text_inversions": {"easynegative": "EasyNegative", "badhandv4": "badhandv4"},
        "model_name": model_name,
        "scheduler": "DPM++2MKarras",
        "guidance_scale": 6.5,
        "num_images_per_prompt": 1,
        "seed": seed,
        "width": 512,
        "height": 512,
        "num_inference_steps": steps,
        "loras": loras,
        "clip_skip": 2
    }
    reply = requests.post(
        "{}/predict".format(SD_ENDPOINT),
        headers={"Content-Type": "application/json"},
        json=sd_request
    )

    img_bytes = b64decode(reply.json()["image_0"])
    cartoonized = Image.open(BytesIO(img_bytes), formats=("png",))

    col2.write("Transformed Image :star2:")
    col2.image(cartoonized)
    st.markdown("\n")
    st.download_button("Download transformed image", convert_image(cartoonized), "cartoonized.png", "cartoonized/png")

st.set_page_config(layout="wide", page_title="Cartoonizer")

st.write("## Cartoonizer - Powered by OctoAI")

st.markdown(
    "The fastest version of Stable Diffusion in the world is now available on OctoAI, where devs run, tune, and scale generative AI models. [Try it for free here.](http://octoml.ai/)"
)

st.markdown(
    "### Upload a photo and turn yourself into a cartoon character!"
)

st.markdown(
    " :camera_with_flash: Tip #1: works best on a square image."
)
st.markdown(
    " :blush: Tip #2: works best on close ups (e.g. portraits), rather than full body or group photos."
)
st.markdown(
    " :woman-getting-haircut: Tip #3: for best results, avoid cropping heads/faces."
)

my_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns(2)

st.button("Generate")
seed = 0
if st.button("Randomize"):
    seed = random.randint(0, 1024)

model_map = {
    "3D Animated A": "cartoon_v2",
    "3D Animated B": "3d_cartoon",
    "3D Animated C": "cartoon",
    "2D Cartoon": "dark-sushi-mix",
    "RPG": "aZovyaRPGArtistTools_v3",
    "Anime": "toonyou_beta3",
}

model = st.selectbox(
    ":lower_left_paintbrush: Style Selector, changes the output style of your image.",
    options=list(model_map.keys())
)

model = model_map[model]

strength = st.slider(
    ":brain: Imagination Slider (lower: closer to original, higher: more imaginative result)",
    3.0, 10.0, 7.0)
    
extra_desc = st.text_input("Add more context to customize the output")
extra_desc_strength = st.slider("Strength of extra context. The higher this is the more your text matters", 1.0, 5.0, value=1.0)
if extra_desc:
    extra_desc = f"({extra_desc}: {extra_desc_strength})"

steps = st.slider(":athletic_shoe: Select the number of steps, more can lead to higher output quality.", 20, 50, value=30)

# Allow lora customization
st.markdown(":test_tube: Try applying these different modifications")

loras = {}
#st
lora_map = {
    "Low Lighting": "LowRA",
    "Simple Animation": "coolkids_v2.5",
    "Pixelated": "pixelart",
    "Pig Tails": "pigtail_hairstyle",
    "Steampunk": "steampunkschematics"
}

for name, lora in lora_map.items():
    selected = st.checkbox(name)
    if selected:
        value = st.slider(f"Select the strength for {name}", 0.0, 2.0)
        loras[lora] = value


st.sidebar.markdown("The image to image generation is achieved via the [following checkpoint](https://civitai.com/models/75650/disney-pixar-cartoon-type-b) on CivitAI.")

st.sidebar.markdown(
    ":warning: **Disclaimer** :warning:: Cartoonizer is built on the foundation of [CLIP Interrogator](https://huggingface.co/spaces/pharma/CLIP-Interrogator) and [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5), and is therefore likely to carry forward the potential dangers inherent in these base models. ***It's capable of generating unintended, unsuitable, offensive, and/or incorrect outputs. We therefore strongly recommend exercising caution and conducting comprehensive assessments before deploying this model into any practical applications.***"
)

st.sidebar.markdown(
    "By releasing this model, we acknowledge the possibility of it being misused. However, we believe that by making such models publicly available, we can encourage the commercial and research communities to delve into the potential risks of generative AI and subsequently, devise improved strategies to lessen these risks in upcoming models. If you are researcher and would like to study this subject further, contact us and we’d love to work with you!"
)

st.sidebar.markdown(
    "Report any issues, bugs, unexpected behaviors [here](https://github.com/tmoreau89/cartoonize/issues)"
)

if my_upload is not None:
    cartoonize_image(my_upload, model, strength, seed, loras, steps, extra_desc)