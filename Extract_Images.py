import tarfile 
import os
from tqdm import tqdm


tar_path = "/data/horse/ws/mosa505e-Multimodal_Rep/data/Breast_Cancer/Croped_Images_3.tar"  # 
out_dir = os.path.join(os.path.dirname(os.path.abspath(tar_path)), "extracted_images")
os.makedirs(out_dir, exist_ok=True)

with tarfile.open(tar_path, "r:*") as tf:
    for member in tqdm(tf.getmembers(), desc="Extracting images"):
        if member.isfile():
            member.name = os.path.basename(member.name)  # flatten to single folder
            tf.extract(member, out_dir)