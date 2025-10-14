# In development

# Photos to Timelapse

It merges multiple photos that should be of the same position considering that the exif data has photo angles and by fixed object selected by user.

# Running

Prepare the environment
```bash
pip3 install -e .
```

And run
```bash
python3 run.py
```

# Next steps

- Parallelize stabilizing method
- Cut selected area in video
- Give option to remove the photo or reselect template for each similarity outlier
- Complete lose parts of images based in other images non zero values, perhaps based in image bright avg 