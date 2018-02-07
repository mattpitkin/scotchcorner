# Singularity

Singularity file for testing build in Python 3.

Build the image with:

```
sudo singularity build --writable python3.img Singularity
```

Run a shell in the container with, e.g.:

```
sudo singularity shell --writable -B /home/matthew/repositories/scotchcorner:/home/scotchcorner python3.img
```
