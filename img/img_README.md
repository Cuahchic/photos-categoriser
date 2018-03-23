# Training Data
In here create a folder called `training` which contains a subfolder for each person you wish to identify.

For example, under `training/Colin` store as many images of the subject Colin you can. Images should only contain the subject in question, i.e. do not include images of more than one person.

# Test Data
In here create a folder called `test` which contains a subfolder for each person you wish to identify.

For example, under `test/Colin` store a bunch of images that were not used to train that contain the subject Colin. It does not matter if there are multiple people in the image, it will attempt to detect whether Colin is in the image.

Not that the `test` folder is optional, if you don't provide one it will just skip this step.

