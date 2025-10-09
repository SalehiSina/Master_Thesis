import torch
import os
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from uni import get_encoder
model, transform = get_encoder(enc_name='uni2-h', device=device)


import numpy as np
import umap
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image

import os

import timm
from huggingface_hub import login, hf_hub_download



login(token="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") # login with your User Access Token, found at https://huggingface.co/settings/tokens

local_dir = "../assets/ckpts/uni2-h/"
os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
hf_hub_download("MahmoodLab/UNI2-h", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
timm_kwargs = {'model_name': 'vit_base_patch16_224',
               'img_size': 224,
               'patch_size': 14,
               'depth': 24,
               'num_heads': 24,
               'init_values': 1e-5,
               'embed_dim': 1536,
               'mlp_ratio': 2.66667*2,
               'num_classes': 0,
               'no_embed_class': True,
               'mlp_layer': timm.layers.SwiGLUPacked,
               'act_layer': torch.nn.SiLU,
               'reg_tokens': 8,
               'dynamic_img_size': True
              }
model = timm.create_model(**timm_kwargs)
model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
model.eval()
model.to(device)
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


if __name__ == "__main__":

    # Path to the images directory
    directory = "../../Data/H&E_Patch_Examples"

    # Get all file names (including directories)
    all_items = os.listdir(directory)

    embedding_dic = {}
    for item in all_items:
        # 1. Load the patch (RGB)
        img = Image.open(f"../../Data/H&E_Patch_Examples/H&E_Patch_Examples/{item}").convert("RGB")

        # 2. Preprocess: resize → tensor → normalize
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        img_tensor = transform(img).unsqueeze(0)  # add batch dim [1, 3, 224, 224]

        # 3. Load UNI model (example, you’d replace with actual checkpoint loading)
        #model = load_uni_model("path/to/uni_checkpoint.pt")  # placeholder
        #model.eval()

        # 4. Extract embedding
        with torch.no_grad():
            embedding = model(img_tensor.to(device))

        center_id = item[28:-13]
        if center_id not in embedding_dic:
            embedding_dic[center_id] = []
            embedding_dic[center_id].append(embedding[0].cpu())
        else:
            embedding_dic[center_id].append(embedding[0].cpu())



    id0 = list(embedding_dic.keys())[0]
    id1 = list(embedding_dic.keys())[1]
    id2 = list(embedding_dic.keys())[2]

    # Combine all vectors into one array
    all_vectors = np.array(embedding_dic[id0] + embedding_dic[id1] + embedding_dic[id2])

    # Create labels (0 for list1, 1 for list2, 2 for list3)
    labels = ([f"region around {id0}"] * len(embedding_dic[id0]) +
            [f"region around {id1}"] * len(embedding_dic[id1]) +
            [f"region around {id2}"] * len(embedding_dic[id2]))


    # Map each string label to a unique integer
    unique_labels = list(set(labels))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    label_colors = [label_to_int[l] for l in labels]

    # Run UMAP (reduce to 2D for plotting)
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(all_vectors)

    # Plot
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(embedding[:,0], embedding[:,1], c=label_colors, cmap="viridis", s=80)

    # Create legend
    handles, _ = scatter.legend_elements()
    unique_labels = list(set(labels))
    plt.legend(handles, unique_labels, title="Groups", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title("UMAP projection of vectors")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.show()