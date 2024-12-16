import tarfile
import os
import pandas as pd

input_tar = "./data/mp-16-images.tar"
output_tar = "./data/neighbourhood_mp-16-images.tar"
text_data = pd.read_csv("./data/MP16_Pro_places365.csv")
curated_img_ids = set(text_data['IMG_ID'].str.strip())

with tarfile.open(input_tar, "r") as src, tarfile.open(output_tar, "w") as dest:
    for member in src:
        if os.path.basename(member.name) in curated_img_ids:
            print("Adding", member.name)
            extracted_file = src.extractfile(member)
            if extracted_file:  # Ensure the file is not None
                dest.addfile(member, extracted_file)

print("Filtered tar file created:", output_tar)