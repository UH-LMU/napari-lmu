# napari-lmu

Scripts to start Napari to browse data from LMU HCS instruments. The data structure is read with Dask, so it will not take huge amounts of memory.

## Prerequisites
On University of Helsinki Windows PCs these can be installed via Software Center.

- Anaconda
- Git

## Create and activate environment
Download the environment file from this repository (env\_napari.yml). Open "Conda Powershell prompt" from Start Menu and run the following command (give the full path to the file):
```
conda env create -f C:\Users\username\Downloads\env_napari.yml
```
You need to do this step only once.

## Activate the enviroment
Open "Conda Powershell Prompt" from Start Menu and run the following command:
```
conda activate napari-lmu
```
This you need to do every time.

## Run the scripts
See builtin help for options for filtering files etc. These can be useful if you have many images but only want to check a few.
```
napari-celliq --help
napari-moldev --help
```
To start, run:

```
napari-celliq
```
or
```
napari-moldev
```

Images are loaded when you select the source folder in the GUI, using the buttons on the right. Progress is shown in the terminal.
