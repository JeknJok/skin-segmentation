#program to remove bg of test imgs
import os
from rembg import remove
from PIL import Image
import io

def remove_bg_and_keep_name(input_path, output_path):
    with open(input_path, 'rb') as i:
        input_data = i.read()

    result = remove(input_data)
    output_image = Image.open(io.BytesIO(result)).convert("RGB")
    output_image.save(output_path)
    print(f"Saved: {output_path}")

# Example
input_folder = 'fullbody/img/'
output_folder = 'no_bg_masks/'

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)  # no _no_bg suffix
        remove_bg_and_keep_name(input_path, output_path)
