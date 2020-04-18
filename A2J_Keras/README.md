**NOTE**
This code is *nearly* functional, but not quite finished. I believe the issue lies in the `post_process` function in `util.py`. In the official A2J code, this post processing is done as part of the loss function, whereas here I tried to make it a part of the model itself. Since some of the ops required are abnormal, the model can be trained, but I've never successfully saved weights to a `.h5` file.  

Additionally, I never implemented a way to load weights into resnet and freeze the model, so *everything* gets trained at once. According to the A2J publication, this isn't ideal.  

Any contribution would be greatly appreciated.
