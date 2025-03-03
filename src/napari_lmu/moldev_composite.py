#!/usr/bin/env python
# coding: utf-8

import click
import dask.array as da
import glob
import matplotlib.pyplot as plt
import napari
import numpy as np
import os
import pandas as pd
import platform
import tifffile
import time
from aicsimageio.aics_image import AICSImage
#from aicsimageio import AICSImage
from magicgui import magicgui
from pathlib import Path
from qtpy.QtWidgets import QVBoxLayout, QWidget, QLabel, QComboBox
from skimage.io import imread, imsave
from tqdm import tqdm

colormap = ["blue", "green", "red"]

PATH = 'Path'
DATE = 'Date'
TSTEP = 'TStep'
ZSTEP = 'ZStep'
PLATE = 'Plate'
WELL = 'Well'
SITE = 'Site'
CHANNEL = 'Channel'
UUID = 'UUID'

def create_file_list(orig, wells=[], nwells=-1, nsites=-1):
    print(orig)
    if not orig:
        return pd.DataFrame()
    
    metadata_columns = {
        'mc2': TSTEP,
        'mc3': ZSTEP,
        'mc4': PLATE,
        'mc5': WELL,
        'mc6': SITE,
        'mc7': CHANNEL,
    }

    files = [(str(x)) for x in orig.glob("**/*.tif") if not "thumb" in x.name]
    df = pd.DataFrame(files, columns=[PATH])

    if not df.empty:
        print(files[-1])

    # Cross-platform pattern with dynamic column names
    pattern = (\
        r'[/\\](?P<{mc4}>[^/\\]*)'\
        + r'(?:[/\\][^/\\]*_Projection)?'\
        + r'(?:[/\\]timepoint\d+)?'\
        + r'[/\\]t(?P<{mc2}>\d+)_(?P<{mc5}>\w\d{{2}})_s(?P<{mc6}>\d{{1,2}})_(?P<{mc7}>w\d)_z(?P<{mc3}>\d+)'\
    ).format(**metadata_columns)

    #print(pattern)
    
    # Apply the regex pattern and extract the desired columns
    df_extracted = df[PATH].str.extract(pattern)
    print()

    # Add the extracted columns back to the original dataframe
    df = df.join(df_extracted)

    df[PLATE] = df[PLATE].astype(str)
    df[WELL] = df[WELL].astype(str)
    df[SITE] = df[SITE].astype(int)
    df[CHANNEL] = df[CHANNEL].astype(str)
    df[TSTEP] = df[TSTEP].astype(int)
    df[ZSTEP] = df[ZSTEP].astype(int)

    if len(wells) > 0:
        mask = df[WELL].isin(wells)
        df = df[mask]
    elif nwells > 0:
        wells = df[WELL].unique()[:nwells]
        mask = df[WELL].isin(wells)
        df = df[mask]
    
    if nsites > 0:
        mask = df[SITE] <= nsites
        df = df[mask]
    
    return df


def get_stacks_and_projections(df):

    # find channels with z-slices
    print(df.ZStep.unique())
    mask = df.ZStep > 1
    ch_z = df[mask][CHANNEL].unique()
    print(f'Channels with slices: {ch_z}')

    # find channels with projections (should be the same as ch_z)
    mask = df.ZStep == 0
    ch_p = df[mask][CHANNEL].unique()
    print(f'Channels with projections: {ch_p}')

    assert (np.sort(ch_z) == np.sort(ch_p)).all()
    
    # separate Z-slices and projection images
    mask_p = df[PATH].str.contains('_Projection/')

    mask_z = df[CHANNEL].isin(ch_z)
    if len(ch_z) == 0:
        mask_z = df[ZSTEP] == 1

    stacks = df[mask_z].copy().reset_index(drop=True)
    projs = df[mask_p].copy().reset_index(drop=True)

    stacks.sort_values(by=[PLATE, WELL, SITE, TSTEP, ZSTEP, CHANNEL], inplace=True, ignore_index=True)
    projs.sort_values(by=[PLATE, WELL, SITE, TSTEP, CHANNEL], inplace=True, ignore_index=True)
    
    print(f'stacks.shape: {stacks.shape}')
    print(f'projs.shape: {projs.shape}')
    
    return stacks, projs


def create_dask_array(grouped2d):
    # Dictionary to store Dask arrays for each plate
    plates = []
    plate_stack = None

    # Prebuild index mapping
    index_map = {}
    index_map_reverse = {}

    # Group by plate and well to handle multiple sites within a well
    for plate, plate_group in grouped2d.groupby(PLATE):
        wells = []

        # Iterate over each well
        for well, well_group in plate_group.groupby(WELL):
            sites = []

            # Iterate over each site
            for site, site_group in tqdm(well_group.groupby(SITE),
                                         desc=f'Reading well {well}'):
                # At this point, we know the plate, well, and site
                # Add an entry to index_mapping for this site
                index_map[(plate, well, site)] = (len(plates), len(wells), len(sites))
                index_map_reverse[(len(plates), len(wells), len(sites))] = (plate, well, site)

                #print(site_group.columns)
                #print(site_group.shape)

                # Explode list columns
                exploded_site_group = site_group.explode([PATH, TSTEP, CHANNEL])
                #print(exploded_site_group.shape)
                #print(exploded_site_group.apply(type).unique())
                #print(exploded_site_group.head())

                channels = []

                # Iterate over each channel and stack them for the current Z-step
                for channel_path in exploded_site_group[PATH]:
                    #print(plate, well, site, channel_path)
                    img = AICSImage(channel_path)
                    # Use img.get_image_dask_data() for lazy loading of data
                    dask_data = img.get_image_dask_data()
                    #print(dask_data.shape)
                    dask_data = dask_data.squeeze()
                    #print(dask_data.shape)
                    channels.append(dask_data)
                #print()
                
                # Stack channels for the current site
                site_stack = da.stack(channels, axis=0)  # Stack Z-slices to form a 3D site-level array
                print(site_stack.shape)
                sites.append(site_stack)

            # Stack all site-level arrays into a well-level array
            well_stack = da.stack(sites, axis=0)  # Stack sites into a well
            wells.append(well_stack)

        # Stack all well-level arrays into a plate-level array
        plate_stack = da.stack(wells, axis=0)  # Stack wells into a plate
        plates.append(plate_stack)

    final_dask_array = da.stack(plates)
    return index_map, index_map_reverse, final_dask_array


def create_dask_array_with_t_z(grouped3d):
    # Dictionary to store Dask arrays for each plate
    plates = []
    plate_stack = None

    # Prebuild index mapping
    index_map = {}
    index_map_reverse = {}

    # Group by plate and well to handle multiple sites within a well
    for plate, plate_group in grouped3d.groupby(PLATE):
        wells = []

        # Iterate over each well
        for well, well_group in plate_group.groupby(WELL):
            sites = []

            # Iterate over each site
            for site, site_group in tqdm(well_group.groupby(SITE),
                                         desc=f'Reading well {well}'):
                # At this point, we know the plate, well, and site
                # Add an entry to index_mapping for this site
                index_map[(plate, well, site)] = (len(plates), len(wells), len(sites))
                index_map_reverse[(len(plates), len(wells), len(sites))] = (plate, well, site)

                #print(site_group.columns)
                #print(site_group.shape)
                #print(site_group[ZSTEP].apply(type).unique())  # Check the type of elements in the ZStep column
                #print(site_group[ZSTEP].head())  # Inspect the first few rows

                t_steps = []

                # Explode both ZStep and Channel columns to ensure they correspond correctly
                exploded_df = site_group.explode([PATH, TSTEP, ZSTEP, CHANNEL])
                #print(f'exploded_df.shape {exploded_df.shape}')
                #print(exploded_df.apply(type).unique())
                #print(exploded_df.head())

                for tstep, tstep_group in exploded_df.groupby(TSTEP):
                    # Group by ZStep to handle stacking of channels for each Z-slice
                    z_steps = []
                    for zstep, zstep_group in tstep_group.groupby(ZSTEP):
                        channels = []

                        # Iterate over each channel and stack them for the current Z-step
                        for channel_path in zstep_group[PATH]:
                            #print(plate, well, site, zstep, channel_path)
                            img = AICSImage(channel_path)
                            # Use img.get_image_dask_data() for lazy loading of data
                            dask_data = img.get_image_dask_data()
                            #print(dask_data.shape)
                            dask_data = dask_data.squeeze()
                            #print(dask_data.shape)
                            channels.append(dask_data)

                        #print()
                        # Stack channels along a new axis (assume channels have same shape)
                        z_step_stack = da.stack(channels, axis=0)  # Stack channels for this Z-step
                        z_steps.append(z_step_stack)
                        
                    t_step_stack = da.stack(z_steps, axis=0)
                    t_steps.append(t_step_stack)

                #print()
                # Stack Z-steps into a full 3D array for the current site
                site_stack = da.stack(t_steps, axis=0)  # Stack Z-slices to form a 3D site-level array
                #print(site_stack.shape)
                sites.append(site_stack)

            # Stack all site-level arrays into a well-level array
            well_stack = da.stack(sites, axis=0)  # Stack sites into a well
            wells.append(well_stack)

        # Stack all well-level arrays into a plate-level array
        plate_stack = da.stack(wells, axis=0)  # Stack wells into a plate
        plates.append(plate_stack)

    final_dask_array = da.stack(plates)
    return index_map, index_map_reverse, final_dask_array


def get_lmu_active1():
    current_os = platform.system()
    
    if current_os == "Windows":
        return "L:\\lmu_active1"
    elif current_os == "Linux":
        return "/mnt/lmu_active1"
    else:
        raise ValueError(f"Unsupported operating system: {current_os}")
    

# Store the last selected folder
last_selected_folder = Path(get_lmu_active1()) / 'instruments/Micro'  # Default to home directory

index_map_reverse_2d = {}
index_map_reverse_3d = {}
final_dask_array_2d = np.array(0)
final_dask_array_3d = np.array(0)


@click.command()
@click.option("--wells", default='ALL', help="Comma-separated list of wells to show.")
@click.option("--nwells", type=int, default=-1, help="Number of wells to show (-1 for all).")
@click.option("--nsites", type=int, default=-1, help="Number of sites per well to show (-1 for all).")
def main(wells, nwells, nsites):

    well_list = wells.split(",") if wells != "ALL" else []
    
    # First create viewer, so it can be used in NavigationWidget
    viewer = napari.Viewer()

    # Create a widget for navigation
    IDX_WELL = 1
    class NavigationWidget(QWidget):
        def __init__(self):
            super().__init__()
            layout = QVBoxLayout()

            # Well selection
            self.well_label = QLabel("Well")
            self.well_combo = QComboBox()
            self.well_combo.currentTextChanged.connect(self.update_image)

            # Adding widgets to layout
            layout.addWidget(self.well_label)
            layout.addWidget(self.well_combo)

            self.setLayout(layout)

            viewer.dims.events.point.connect(self._update_display)

        def update_wells(self, wells):
            self.wells = wells
            self.well_combo.clear()
            self.well_combo.addItems(wells)
            #self.well_selector.well.value = wells[0] if wells else None

        def _update_display(self):
            #print("_update_display")
            slider_index = viewer.dims.point[IDX_WELL]
            slider_index = round(slider_index)
            self.well_combo.setCurrentText(self.wells[slider_index])

        def update_image(self):
            well = self.well_combo.currentText()
            #print(well)  # Debugging print

            # Select data based on plate, well, and site
            if well in self.wells:
                viewer.dims.set_point(IDX_WELL, self.wells.index(well))

    navigate_wells = NavigationWidget()


    @magicgui(
        folder={"label": "Select Folder", "mode": "d", "value": last_selected_folder},  # "d" stands for directory
        auto_call=True,
    )
    def select_folder(folder: Path):
        if not folder or not folder.exists():
            print("Invalid folder")
            return

        global last_selected_folder
        last_selected_folder = folder  # Store the selected folder

        global index_map_reverse_2d
        global index_map_reverse_3d
        global final_dask_array_2d
        global final_dask_array_3d

        # create data frame with file list
        df = create_file_list(folder, wells=well_list, nwells=nwells, nsites=nsites)
        
        # split file list to stacks and projections
        stacks, projs = get_stacks_and_projections(df)

        # read wavelengths from file list
        wavelengths = df.Channel.unique()
        wavelengths = sorted(wavelengths)

        viewer.layers.clear()  # Clear existing layers

        if not stacks.empty:
            t = time.time()
            grouped3d = stacks.groupby(by=[PLATE, WELL, SITE]).agg(list)
            index_map_3d, index_map_reverse_3d, final_dask_array_3d = create_dask_array_with_t_z(grouped3d)
            elapsed = time.time() - t
            print(f'Read stacks in {elapsed}')
            
            n_zsteps = final_dask_array_3d.shape[3]

            plates = list(stacks[PLATE].unique())
            wells = list(stacks[WELL].unique())
            sites = list(stacks[SITE].unique())

            # Update the well dropdown
            navigate_wells.update_wells(wells)

            # Add 3D image with ZStep axis
            viewer.add_image(
                final_dask_array_3d, 
                channel_axis=5,  # Channel is 4th dimension in 3D
                name=wavelengths,
            )

            if not projs.empty:
                t = time.time()
                grouped_projs = projs.groupby(by=[PLATE, WELL, SITE]).agg(list)
                index_map_2d, index_map_reverse_2d, final_dask_array_2d = create_dask_array(grouped_projs)
                elapsed = time.time() - t
                print(f'Read projections in {elapsed}')
                
                # Expand 2D projection image to match the Z-axis length of the 3D image
                expanded_2d_da = da.repeat(final_dask_array_2d[:, :, :, None, :, :, :], repeats=n_zsteps, axis=3)

                names_2d = [w + " projection" for w in wavelengths]
                viewer.add_image(
                    expanded_2d_da, 
                    channel_axis=5,  # Channel is 4th dimension in 3D
                    name=names_2d,
                )

            viewer.dims.axis_labels = ['Plate', 'Well', 'Site', 'TimeStep', 'Z-slice', 'X', 'Y']

            # start from Z-slice 0 to have labels visible
            # start from well 0 to match pull-down
            # start from site 0
            for i in range(len(viewer.dims.point)):
                #print(i)
                viewer.dims.set_point(i,0)

        else:
            print('No images found (stacks.empty)')


    viewer.window.add_dock_widget(select_folder)
    viewer.window.add_dock_widget(navigate_wells)


    def map_index_to_plate_well_site(index):
        #print(f'map_index_to_filename {index}')
        try:
            # Extract Plate, Well, and Site from the Napari index
            plate_idx, well_idx, site_idx, path_idx, _, _ = index  # Ignore last two indices

            plate, well, site = index_map_reverse_3d[(plate_idx, well_idx, site_idx)]
            #print(f'plate, well, site: {plate}, {well}, {site}')

            return plate, well, site

        except (KeyError, IndexError):
            return "Unknown plate or wellsite index"


    # Define a function to save the current view
    def save_current_view():
        # Get current index from Napari sliders
        current_index = tuple(viewer.dims.current_step)

        # Map index to image slice
        img = final_dask_array_3d[current_index[:3]]  # Adjust indexing based on shape
        print(img.shape)

        plate, well, site = map_index_to_plate_well_site(current_index)

        # Generate a filename using index
        filename = f"saved_view_{'_'.join([plate, well, str(site)])}.tif"

        # Choose save directory
        save_dir = Path.home() / "Napari_Saved_Views"
        save_dir.mkdir(exist_ok=True)  # Create directory if needed
        save_path = save_dir / filename

        # Save image at full resolution
        tifffile.imwrite(save_path, img, photometric='minisblack')

        print(f"Saved: {save_path}")

    # Create a Napari button widget
    save_button = magicgui(save_current_view, call_button="Save Current View")

    # Add button to Napari viewer
    viewer.window.add_dock_widget(save_button, area="right")

    viewer.window._qt_window.setWindowTitle("napari-moldev")

    napari.run()


if __name__ == "__main__":
    main()

