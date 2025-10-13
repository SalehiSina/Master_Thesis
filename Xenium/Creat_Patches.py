import numpy as np
import random
import spatialdata as sd
#import spatialdata_plot
#from pathlib import Path
#from spatialdata_io import xenium

#from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
#import seaborn as sns



path_write = "../../Data/hPancreas_Cancer_zarr/data.zarr"
sdata = sd.read_zarr(path_write)


cropped_sdata_dic = {}
id_lists = []
for i in tqdm(range(0,3)):

  # Randomly select a cell and print its spatial coordinates
  random_cell = random.choice(sdata.tables['table'].obs.index)
  random_cell_id = sdata.tables['table'].obs['cell_id'][random_cell]
  id_lists.append(random_cell_id)
  print(f'Randomly selected cell: {random_cell_id}')

  # Transform the cell boundaries to global coordinates
  navi = sd.transform(
      sdata.shapes['cell_boundaries'],
      to_coordinate_system="global"
  )

  cell_center = navi.centroid[random_cell_id]
  print(f'Cell center in global coordinates: {cell_center}')

  # Access aligned H&E image in global coords
  he_img_global = sd.transform(
      sdata.images['he_image'],
      to_coordinate_system="global"
  )

  # Get cell center (yx format for plotting)
  x, y = cell_center.x, cell_center.y

  # Define crop size (e.g. 200 microns around the cell)
  half_size = 164.5*4  # adjust depending on how much context you want
  xmin, xmax = x - half_size, x + half_size
  ymin, ymax = y - half_size, y + half_size

  # Use bounding_box_query to get the cropped SpatialData
  sdata_crop = sd.bounding_box_query(
      sdata,
      min_coordinate=[xmin, ymin],
      max_coordinate=[xmax, ymax],
      axes=("x", "y"),
      target_coordinate_system="global"
  )
  cropped_sdata_dic[random_cell_id] = sdata_crop

import warnings
warnings.filterwarnings('ignore')


Patches_dic = {}
for id in tqdm(id_lists):

  f = True
  i = 0
  while f:

    # Randomly select a cell and print its spatial coordinates
    random_cell = random.choice(cropped_sdata_dic[id].tables['table'].obs.index)
    random_cell_id = cropped_sdata_dic[id].tables['table'].obs['cell_id'][random_cell]
    #print(f'Randomly selected cell {random_cell_id} inside the frame with center cell {id}')

    # Transform the cell boundaries to global coordinates
    navi = sd.transform(
        cropped_sdata_dic[id].shapes['cell_boundaries'],
        to_coordinate_system="global"
    )

    cell_center = navi.centroid[random_cell_id]
    #print(f'Cell center in global coordinates: {cell_center}')

    # Get cell center (yx format for plotting)
    x, y = cell_center.x, cell_center.y

    # Define crop size (e.g. 200 microns around the cell)
    half_size = 164.5  # adjust depending on how much context you want
    xmin, xmax = x - half_size, x + half_size
    ymin, ymax = y - half_size, y + half_size

    # Use bounding_box_query to get the cropped SpatialData
    Patches_dic[f'area_around_{id}_center_{i}'] = sd.bounding_box_query(
        cropped_sdata_dic[id],
        min_coordinate=[xmin, ymin],
        max_coordinate=[xmax, ymax],
        axes=("x", "y"),
        target_coordinate_system="global"
    )
    if Patches_dic[f'area_around_{id}_center_{i}'].images["he_image"]['scale0'].image.data.compute().shape == (3,512,512):
      print("correct size")
      print("i = ",i)
      i = i+1
    else:
      print("wrong size")
      print(Patches_dic[f'area_around_{id}_center_{i}'].images["he_image"]['scale0'].image.data.compute().shape)
      print("i = ",i)

    if i == 9:
      f = False

for id in tqdm(id_lists):
  for i in tqdm(range(0,9)):
    # Get the cropped H&E image (already aligned)
    he_crop = Patches_dic[f'area_around_{id}_center_{i}'].images["he_image"]

    # Convert to numpy
    he_crop_array = he_crop['scale0'].image.data.compute()
    he_crop_array = he_crop_array.transpose(1, 2, 0)  # (cyx) → (yx, x, c)

    normalized_he = (he_crop_array - np.min(he_crop_array)) / (np.max(he_crop_array) - np.min(he_crop_array))
    plt.imsave(f"/../../Data/H&E_Patch_Examples_1/XeniumH&E_patch_area_around_{id}_center_{i}.png", normalized_he)
