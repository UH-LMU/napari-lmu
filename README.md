# napari-lmu

Scripts to start Napari to browse data from LMU HCS instruments. The data structure is read with Dask, so it will not take huge amounts of memory.

## Prerequisites
On University of Helsinki Windows PCs these can be installed via Software Center.

- Anaconda
- Git

## Create and activate environment
Download the environment file from this repository (env\_napari.yml).
```
conda env create -f env_napari.yml
conda activate napari-lmu
```

## Run the scripts
See builtin help for options for filtering files etc.
```
napari-celliq --help
napari-moldev --help
```
Images are loaded when you select the source folder in the GUI. Progress is shown in the terminal.
