#!/usr/bin/env python
# coding: utf-8

import dask
import dask.array as da
import glob
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import napari
import numpy as np
import os
import pandas as pd
import platform
import time
from aicsimageio.aics_image import AICSImage
#from aicsimageio import AICSImage
from magicgui import magicgui
from pathlib import Path
from qtpy.QtWidgets import QVBoxLayout, QWidget, QLabel, QComboBox
from skimage.io import imread
from tqdm import tqdm

PATH = 'Path'
DIR = 'Directory'
DATE = 'Date'
TIMEPOINT = 'TimePoint'
ZSTEP = 'ZStep'
PLATE = 'Plate'
WELL = 'Well'
SITE = 'Site'
WELLSITE = 'WellSite'
CHANNEL = 'Channel'
UUID = 'UUID'

metadata_columns = {
    'mc2': TIMEPOINT,
    'mc3': ZSTEP,
    'mc4': PLATE,
    'mc5': WELL,
    'mc6': SITE,
    'mc7': CHANNEL,
    'mc8': WELLSITE,
}

def create_file_list(orig, ftype='tif', max_sites=-1):
    print(orig)
    if not orig:
        return pd.DataFrame()
    
    files = [(str(x)) for x in orig.glob(f"*/*.{ftype}") if not "_flows" in x.name]
    df = pd.DataFrame(files, columns=[PATH])

    if not df.empty:
        print(files[-1])

        
    # Cross-platform pattern with dynamic column names
    pattern = (\
        r'[/\\](?P<{mc4}>[^/\\]*)'\
        r'[/\\]Well[ _](?P<{mc8}>[A-Z]\d*_\d*)'\
        + r'[/\\](?P<{mc2}>\d\d\d\d_\d\d_\d\d_\d\d_\d\d_\d\d_\d\d).*[tif,jpg,png]'\
    ).format(**metadata_columns)

    #print(pattern)
    
    # Apply the regex pattern and extract the desired columns
    df_extracted = df[PATH].str.extract(pattern)
    #print()

    # Add the extracted columns back to the original dataframe
    df = df.join(df_extracted)
    
    #print(df.head(1))
    df[DIR] = df[PATH].apply(lambda x: str(Path(x).parent))
    df[PLATE] = df[PLATE].astype(str)
    df[WELLSITE] = df[WELLSITE].astype(str)
    #df[WELL] = df[WELL].astype(str)
    #df[SITE] = df[SITE].astype(int)
    #df[CHANNEL] = 1
    #df[ZSTEP] = 1

    if max_sites > 0:
        mask = df[SITE] <= max_sites
        df = df[mask]

    return df


pattern = (\
    r'[/\\](?P<{mc4}>[^/\\]*)'\
    r'[/\\]Well[ _](?P<{mc8}>[A-Z]\d*_\d*)'\
    + r'[/\\](?P<{mc2}>\d\d\d\d_\d\d_\d\d_\d\d_\d\d_\d\d_\d\d).*[tif,jpg,png]'\
).format(**metadata_columns)

import re
regex = re.compile(pattern)
example = '/home/hajaalin/data/user/test/Well_A2_02/2025_02_11_15_29_31_65.tif'

search = regex.search(example)
search.groups()


def create_dask_array(grouped2d):
    # Dictionary to store Dask arrays for each plate
    plates = []
    plate_stack = None

    # Prebuild index mapping
    index_map = {}
    index_map_reverse = {}

    # Group by plate and well to handle multiple sites within a well
    for plate, plate_group in grouped2d.groupby(PLATE):
        wellsites = []

        # Iterate over each well
        for wellsite, wellsite_group in plate_group.groupby(WELLSITE):
            # At this point, we know the plate, well, and site
            # Add an entry to index_mapping for this site
            index_map[(plate, wellsite)] = (len(plates), len(wellsites))
            index_map_reverse[(len(plates), len(wellsites))] = (plate, wellsite)

            #print('site_group')
            #print(site_group.columns)
            #print(site_group.shape)

            # Explode list columns
            exploded_site_group = wellsite_group.explode([PATH])
            #print('exploded_site_group')
            #print(exploded_site_group.shape)
            #print(exploded_site_group.apply(type).unique())
            #print(exploded_site_group.head())

            channels = []

            # Iterate over each channel and stack them for the current Z-step
            for channel_path in tqdm(exploded_site_group[PATH], desc='Reading site ' + wellsite):
                #print(plate, wellsite, channel_path)
                img = AICSImage(channel_path)
                # Use img.get_image_dask_data() for lazy loading of data
                dask_data = img.get_image_dask_data()
                #print(dask_data.shape)
                dask_data = dask_data.squeeze()
                #print(dask_data.shape)
                channels.append(dask_data)
            #print()

            # Stack channels for the current site
            wellsite_stack = da.stack(channels, axis=0)  # Stack Z-slices to form a 3D site-level array
            #print(site_stack.shape)
            wellsites.append(wellsite_stack)
            
            
        # Stack all well-level arrays into a plate-level array
        plate_stack = da.stack(wellsites, axis=0)  # Stack wellsites into a plate
        plates.append(plate_stack)

    final_dask_array = da.stack(plates)
    return index_map, index_map_reverse, final_dask_array


def save_empty_label_image(folder, timestamp, extension, shape=(1040, 1392)):
    # Create an empty (black) image
    image = np.zeros(shape, dtype=np.uint8)  # uint8 ensures valid pixel values (0-255)

    path = str(Path(folder) / (timestamp + extension))
    #print(path)
    
    # Convert to PIL Image and save
    imageio.imwrite(path, image)
    
    
def get_lmu_active1():
    current_os = platform.system()
    
    if current_os == "Windows":
        return "L:\\lmu_active1"
    elif current_os == "Linux":
        return "/mnt/lmu_active1"
    else:
        raise ValueError(f"Unsupported operating system: {current_os}")


    
# Store the last selected folder
last_selected_folder = Path('E:\LocalData')  
last_selected_folder_labels = Path('E:\LocalData') 

# Store dataframe with images
df_images = pd.DataFrame()
df_images_grouped = pd.DataFrame()
df_labels = pd.DataFrame()
df_missing_labels = pd.DataFrame()
index_map = {}
index_map_reverse = {}


def main():
    
    # First create viewer, so it can be used in NavigationWidget
    viewer = napari.Viewer()

    # Create a widget for navigation
    IDX_WELLSITE = 1
    class NavigationWidget(QWidget):
        def __init__(self):
            super().__init__()
            layout = QVBoxLayout()
            
            # Well selection
            self.wellsite_label = QLabel("WellSite")
            self.wellsite_combo = QComboBox()
            self.wellsite_combo.currentTextChanged.connect(self.update_image)

            # Adding widgets to layout
            layout.addWidget(self.wellsite_label)
            layout.addWidget(self.wellsite_combo)

            self.setLayout(layout)
            
            viewer.dims.events.point.connect(self._update_display)
            
        def update_wellsites(self, wellsites):
            self.wellsites = wellsites
            self.wellsite_combo.clear()
            self.wellsite_combo.addItems(wellsites)
            
        def _update_display(self):
            #print("_update_display")
            slider_index = viewer.dims.point[IDX_WELLSITE]
            slider_index = round(slider_index)
            self.wellsite_combo.setCurrentText(self.wellsites[slider_index])

        def update_image(self):
            wellsite = self.wellsite_combo.currentText()
            #print(well)  # Debugging print

            # Select data based on plate, well, and site
            if wellsite in self.wellsites:
                viewer.dims.set_point(IDX_WELLSITE, self.wellsites.index(wellsite))

    navigate_wellsites = NavigationWidget()

        
    def map_index_to_filename(index):
        #print(f'map_index_to_filename {index}')
        try:
            # Extract Plate, Well, and Site from the Napari index
            plate_idx, wellsite_idx, path_idx, _, _ = index  # Ignore last two indices

            plate, wellsite = index_map_reverse[(plate_idx, wellsite_idx)]
            #print(f'plate, well, site: {plate}, {well}, {site}')

            # Find the row using MultiIndex lookup
            row = df_images_grouped.loc[(plate, wellsite)]

            # Get the Path list from the dataframe
            path_list = row[PATH]
            #print(f'{path_list}')

            # Select the correct Timepoint using time_idx
            selected_path = path_list[path_idx]
            #print(f'{selected_path}')

            return selected_path

        except (KeyError, IndexError):
            return "Unknown file"
            
        except TypeError:
            print(index_map)
            print(index_map_reverse)
            print(index)


    def update_filename(event):
        current_index = tuple(viewer.dims.current_step)  # Get current slider positions
        #print(f"Current index: {current_index}")

        # Map index to filename using your dataframe
        filename = map_index_to_filename(current_index)  # Define this function

        # Display filename
        viewer.text_overlay.visible = True
        viewer.text_overlay.text = filename  # Simple overlay in viewer


    @magicgui(
        folder={"label": "Select folder (images)", "mode": "d", "value": last_selected_folder},  # "d" stands for directory
        auto_call=True,
    )
    def select_folder_images(folder: Path):
        if not folder or not folder.exists():
            print("Invalid folder")
            return
        
        last_selected_folder = folder  # Store the selected folder
        
        # create data frame with file list
        df = create_file_list(folder)
        df.sort_values(by=[PLATE, WELLSITE, TIMEPOINT], inplace=True, ignore_index=True)
        
        global df_images
        global df_images_grouped
        df_images = df.copy()
        global index_map
        global index_map_reverse
        
        viewer.layers.clear()  # Clear existing layers

        if not df.empty:
            t1 = time.time()
            plates = list(df[PLATE].unique())
            wellsites = list(df[WELLSITE].unique())
            
            # Update the well dropdown
            navigate_wellsites.update_wellsites(wellsites)

            grouped_df = df.groupby(by=[PLATE, WELLSITE]).agg(list)
            index_map, index_map_reverse, final_dask_array_2d = create_dask_array(grouped_df)
            
            # using multiple workers to build the dask array just seemed to slow things down...
            #index_map, index_map_reverse, final_dask_array_2d = parallel_dask_array(df)

            elapsed = time.time() - t1
            print(f'Read images in {elapsed}')
            
            df_images_grouped = grouped_df.copy()

            viewer.add_image(
                final_dask_array_2d, 
                name = 'original',
            )

            viewer.dims.axis_labels = ['WellSite', 'Timepoint', 'X', 'Y']

            # start from Z-slice 0 to have labels visible
            # start from well 0 to match pull-down
            # start from site 0
            for i in range(len(viewer.dims.point)):
                #print(i)
                viewer.dims.set_point(i,0)
                
            # Connect to Napari slider updates
            viewer.dims.events.current_step.connect(update_filename)

        else:
            print('No images found (stacks.empty)')



    @magicgui(
        folder={"label": "Select folder (labels)", "mode": "d", "value": last_selected_folder_labels},  # "d" stands for directory
        auto_call=True,
    )
    def select_folder_labels(folder: Path):
        if not folder or not folder.exists():
            print("Invalid folder")
            return
        
        last_selected_folder_labels = folder  # Store the selected folder
        
        # create data frame with file list
        df = create_file_list(folder, ftype='png')
        df.sort_values(by=[PLATE, WELLSITE, TIMEPOINT], inplace=True, ignore_index=True)
        
        global df_labels
        global df_missing_labels
        df_labels = df.copy()

        #print(f'index_map {index_map}')
        #print(f'index_map_reverse {index_map_reverse}')
        #print(df.columns)
        #print(df_images[WELLSITE].unique())
        #print(df_labels[WELLSITE].unique())
        #print(df_images[TIMEPOINT].unique())
        #print(df_labels[TIMEPOINT].unique())

        # find images without labels
        diff = df_images.merge(df, on=[WELLSITE, TIMEPOINT], how='left', indicator=True, suffixes=('', '_labels'))
        df_missing_labels = diff[diff['_merge'] == 'left_only'].drop(columns=['_merge'])
            
        if not df_missing_labels.empty:
            print('missing labels')
            print(df_missing_labels.columns)
            print(df_missing_labels[[WELLSITE, TIMEPOINT]])
            
            print('adding empty label images')
            for index, row in df_missing_labels.iterrows():
                folder = None
                mask = df[WELLSITE] == row[WELLSITE]
                folder = df[mask][DIR].unique()[0]
                if folder:
                    save_empty_label_image(folder, row[TIMEPOINT], '_dummy_labels.png')
                    new_row = {PLATE: row[PLATE], WELLSITE: row[WELLSITE], TIMEPOINT: row[TIMEPOINT],\
                               PATH: folder}
                    df_labels.loc[len(df_labels)] = new_row
                    df_labels.sort_values(by=[PLATE, WELLSITE, TIMEPOINT], inplace=True, ignore_index=True)
                else:
                    print('No labels found for ' + folder)
                    return
            return
            
        
        # remove previous label layer
        if len(viewer.layers) > 1:
            viewer.layers.pop()

        if not df.empty:
            plates = list(df[PLATE].unique())
            wellsites = list(df[WELLSITE].unique())
            
            # Update the well dropdown
            #navigate_wells.update_wells(wells)

            grouped_df = df.groupby(by=[PLATE, WELLSITE]).agg(list)
            _index_map, _index_map_reverse, final_dask_array_2d = create_dask_array(grouped_df)
            
            viewer.add_labels(
                final_dask_array_2d, 
                name = 'cellpose',
            )

            # start from Z-slice 0 to have labels visible
            # start from well 0 to match pull-down
            # start from site 0
            for i in range(len(viewer.dims.point)):
                #print(i)
                viewer.dims.set_point(i,0)
                
        else:
            print('No label images found (stacks.empty)')

    # Create a container widget to hold the folder selectors
    class FolderSelectors(QWidget):
        def __init__(self):
            super().__init__()
            layout = QVBoxLayout()
            layout.addWidget(select_folder_images.native)
            layout.addWidget(select_folder_labels.native)
            layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
            layout.setSpacing(5)  # Adjust spacing
            self.setLayout(layout)

    #viewer.window.add_dock_widget(select_folder_images, area='right')
    #viewer.window.add_dock_widget(select_folder_labels, area='right')
    viewer.window.add_dock_widget(FolderSelectors(), area='right')
    viewer.window.add_dock_widget(navigate_wellsites, area='right')
    

    napari.run()


if __name__ == "__main__":
    main()



