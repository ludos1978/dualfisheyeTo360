# dualfisheyeTo360
python script to convert a dual fisheye image from a yi 360 to a equirectangular image
It does a pretty good job in my opinion and also aligns colors of the both fisheyes

# usage
- edit variables at ~line 170

### values of the camera, can be modified in the pygame interface

    lVertOffset = 30
    fov = 199.2
    rYaw = 91.2
    rPitch = 0.0
    rRoll = 1.9
    left_color_overlap = True # use the left or the right side of the overlapping parts of the image to do color matching


## In the pygame interface use
- w,s: Roll control
- a,d: Yaw control
- q,e: Pitch control
- r,f: field of view control
- i,k: fisheye lens vertical offset control
- x: finish editing the values and generate the image (original location +"-merged.png")
-  -> also check the console output for the values to copy into the script for default values

# requirements
- see imports
- also included in this repo https://github.com/cynricfu/multi-band-blending
