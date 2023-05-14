import os

try:
    import ffmpeg
    import pygame
    import numpy as np
    import cv2
    from PIL import Image
    import code
    import cv2.xphoto
    import multi_band_blending
except:
    # if not available install packages
    import pip
    pip.main(['install', 'ffmpeg-python'])
    pip.main(['install', 'opencv-contrib-python'])

    import ffmpeg
    import pygame
    import numpy as np
    import cv2
    from PIL import Image
    import code
    import cv2.xphoto
    import multi_band_blending

def saveNumpyImage (npImage, filename):
    # save the image for testing the conversion
    im = Image.fromarray(npImage)
    im.save(filename)

def saveFfmpegImage (ffmpegImage, filename):
    ffmpegImage.output(filename, loglevel="quiet").overwrite_output().run()

def loadFfmpegImage (filename):
    return ffmpeg.input(filename)

def ffmpegToNpImage(ffmpegImage, imgHeight, imgWidth):
    buffer, err = (ffmpeg.output(ffmpegImage, "pipe:", vframes=1, format='rawvideo', pix_fmt="rgb24", loglevel="quiet").run(capture_stdout=True))
    return np.frombuffer(buffer, np.uint8).reshape([imgHeight, imgWidth, 3])

def ffmpegToPygameImage(ffmpegImage, imgHeight, imgWidth):
    ffmpegImageString, err = ffmpeg.output(ffmpegImage, "pipe:", format="rawvideo", pix_fmt="rgb24", loglevel="quiet").run(capture_stdout=True)
    return pygame.image.fromstring(ffmpegImageString, [imgWidth, imgHeight], "RGB")


# --- Gray world ---
# The Gray World algorithm assumes that the average color of a scene under standard lighting conditions should appear gray. It scales the color channels to achieve a balanced color distribution.
def gray_world(image):
    # Calculate average channel values
    avg_r = np.mean(image[:, :, 0])
    avg_g = np.mean(image[:, :, 1])
    avg_b = np.mean(image[:, :, 2])

    # Compute scaling factors
    scale_r = avg_g / avg_r
    scale_b = avg_g / avg_b

    # Apply scaling to each channel
    corrected_image = image.copy()
    corrected_image[:, :, 0] = np.clip(image[:, :, 0] * scale_r, 0, 255)
    corrected_image[:, :, 2] = np.clip(image[:, :, 2] * scale_b, 0, 255)

    return corrected_image
# --- Gray world end ---

# --- white patch ---
# The White Patch algorithm assumes that the brightest point in an image should correspond to the white reference. It scales the color channels based on the maximum channel value.
def white_patch(image):
    # Find maximum channel values
    max_r = np.max(image[:, :, 0])
    max_g = np.max(image[:, :, 1])
    max_b = np.max(image[:, :, 2])

    # Compute scaling factors
    scale_r = 255 / max_r
    scale_g = 255 / max_g
    scale_b = 255 / max_b

    # Apply scaling to each channel
    corrected_image = image.copy()
    corrected_image[:, :, 0] = np.clip(image[:, :, 0] * scale_r, 0, 255)
    corrected_image[:, :, 1] = np.clip(image[:, :, 1] * scale_g, 0, 255)
    corrected_image[:, :, 2] = np.clip(image[:, :, 2] * scale_b, 0, 255)

    return corrected_image
# --- white patch end ---

# --- NumPy White Balance ---
# https://stackoverflow.com/questions/1175393/white-balance-algorithm
# white balance for every channel independently
def numpyWhitebalance(channel, perc = 0.05):
    mi, ma = (np.percentile(channel, perc), np.percentile(channel,100.0-perc))
    channel = np.uint8(np.clip((channel-mi)*255.0/(ma-mi), 0, 255))
    return channel
# image = cv2.imread("foo.jpg", 1) # load color
# imWB  = np.dstack([wb(channel, 0.05) for channel in cv2.split(img)] )
# --- NumPy White Balance END ---

def numpySimpleWhitebalance (image):
    wb = cv2.xphoto.createSimpleWB()
    return wb.balanceWhite(image)


# --- blending mask generation ---
def midNpBlendMask(height, width):
    mask = np.zeros((height, width), dtype=np.float32)
    # Define the region of img1 that should be visible
    mask[:, width*1//4:width*3//4] = 1  # Left half of the mask is white
    return cv2.merge([mask, mask, mask])
# --- blending masks end ---

# --- 
# Define a function that calculates the color correction parameters
def calculate_correction_params(img1, img2):
    # Calculate the mean and standard deviation of each color channel for both images
    mean1, std1 = cv2.meanStdDev(img1)
    mean2, std2 = cv2.meanStdDev(img2)

    # Flatten the results and convert to float64
    mean1 = mean1.ravel().astype(np.float64)
    std1 = std1.ravel().astype(np.float64)
    mean2 = mean2.ravel().astype(np.float64)
    std2 = std2.ravel().astype(np.float64)

    # Calculate the gain and bias for each color channel
    gain = std2 / std1
    bias = mean2 - gain * mean1

    return gain, bias

# Define a function that applies the color correction parameters
def apply_correction_params(img, gain, bias):
    # Ensure that gain and bias are the correct shape
    gain = gain[:, np.newaxis, np.newaxis]
    bias = bias[:, np.newaxis, np.newaxis]

    # Ensure that img, gain, and bias are all float64
    img = np.float64(img)
    # Reshape gain and bias to be single-row, single-channel arrays
    gain = gain.reshape(1, -1).astype(np.float64)
    bias = bias.reshape(1, -1).astype(np.float64)

    print (f"gain {gain}")
    print (f"bias {bias}")

    # Apply the gain and bias to each color channel
    corrected_img = cv2.multiply(img, gain)
    corrected_img = cv2.add(corrected_img, bias)

    # Convert the corrected image back to 8-bit unsigned integer
    corrected_img = np.clip(corrected_img, 0, 255).astype('uint8')

    return corrected_img

def buildFilepath(customLabel=""):
    global image_filename, image_filenameNoext, image_onlypath

    outpath = f"{image_filenameNoext}{customLabel}.png"
    if image_onlypath != "":
        outpath = f"{image_onlypath}{os.path.sep}"+outpath
    
    return outpath

try:
    image_filepath    = "Orig/Spanien_202003_360/YIVR_C0792003_0794_360.JPG"

    # values of the camera, can be modified in the pygame interface
    lVertOffset = 30
    fov = 199.2
    rYaw = 91.2
    rPitch = 0.0
    rRoll = 1.9
    left_color_overlap = True # use the left or the right side of the overlapping parts of the image to do color matching



    # determine file path informations
    image_filename = os.path.basename(image_filepath)
    image_filenameNoext = os.path.splitext(image_filename)[0]
    image_onlypath = os.path.dirname(image_filepath)

    # Read the input image
    image = loadFfmpegImage(image_filepath)

    # Get the dimensions of the image
    probe = ffmpeg.probe(image_filepath)
    origImgWidth = probe["streams"][0]["width"]
    origImgHeight = probe["streams"][0]["height"]

    # should be fixed
    lYaw = 90
    lPitch = 0
    lRoll = 0

    lIhFov = fov
    lIvFov = fov
    rIhFov = fov
    rIvFov = fov

    pygame.init()
    screen_width, screen_height = 1600, 800
    screen = pygame.display.set_mode((screen_width, screen_height))

    running = True
    update = True
    while running:
        pygame.event.get()
        keys = pygame.key.get_pressed()
        # for event in pygame.event.get():
        if keys[pygame.K_x]:
            running = False
        speed = 0.1
        if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]):
            speed = 1.0
        # yaw
        if keys[pygame.K_a]:
            rYaw += speed
            update = True
        if keys[pygame.K_d]:
            rYaw -= speed
            update = True
        # roll
        if keys[pygame.K_w]:
            rRoll += speed
            update = True
        if keys[pygame.K_s]:
            rRoll -= speed
            update = True
        # pitch
        if keys[pygame.K_q]:
            rPitch += speed
            update = True
        if keys[pygame.K_e]:
            rPitch -= speed
            update = True
        # fov
        if keys[pygame.K_f]:
            rIhFov -= speed
            rIvFov -= speed
            lIhFov -= speed
            lIvFov -= speed
            update = True
        if keys[pygame.K_r]:
            rIhFov += speed
            rIvFov += speed
            lIhFov += speed
            lIvFov += speed
            update = True
        # offset
        if keys[pygame.K_i]:
            lVertOffset -= speed * 10
            update = True
        if keys[pygame.K_k]:
            lVertOffset += speed * 10
            update = True
        
        if update:
            print (f"offset: {lVertOffset:.1f}")
            print (f"l: {lIhFov:.1f}, {lIvFov:.1f}, {lYaw:.1f}, {lPitch:.1f}, {lRoll:.1f}")
            print (f"r: {rIhFov:.1f}, {rIvFov:.1f}, {rYaw:.1f}, {rPitch:.1f}, {rRoll:.1f}")

            lFishImg = image.crop(0, 0 + lVertOffset, origImgWidth, origImgHeight // 2)
            rFishImg = image.crop(0, origImgHeight // 2 - 0, origImgWidth, origImgHeight // 2)
            
            lEquiImg = lFishImg.filter_("v360", "fisheye", "e", ih_fov=lIhFov, iv_fov=lIvFov, yaw=lYaw, pitch=lPitch, roll=lRoll).filter("scroll", hpos=0.25)
            rEquiImg = rFishImg.filter_("v360", "fisheye", "e", ih_fov=rIhFov, iv_fov=rIvFov, yaw=rYaw, pitch=rPitch, roll=rRoll).filter("scroll", hpos=0.75)

            pyImgL = ffmpegToPygameImage(lEquiImg, 2880, 5760).convert_alpha()
            pyImgL.fill((255, 255, 255, 128), special_flags=pygame.BLEND_RGBA_MULT)
            pyImgR = ffmpegToPygameImage(rEquiImg, 2880, 5760).convert_alpha()
            pyImgR.fill((255, 255, 255, 128), special_flags=pygame.BLEND_RGBA_MULT)

            update = False

            scaledImgL = pygame.transform.scale(pyImgL, (screen_width, screen_height))
            scaledImgR = pygame.transform.scale(pyImgR, (screen_width, screen_height))
        
        screen.blit(scaledImgL, (0, 0))
        screen.blit(scaledImgR, (0, 0))
        pygame.display.update()
    pygame.quit()


    # --- generate merged image ---

    lEquiImg = lFishImg.filter_("v360", "fisheye", "e", ih_fov=lIhFov, iv_fov=lIvFov, yaw=lYaw, pitch=lPitch, roll=lRoll).filter("scroll", hpos=0.25)
    rEquiImg = rFishImg.filter_("v360", "fisheye", "e", ih_fov=rIhFov, iv_fov=rIvFov, yaw=rYaw, pitch=rPitch, roll=rRoll).filter("scroll", hpos=0.75)

    lNpImg = ffmpegToNpImage(lEquiImg, 2880, 2880*2)
    rNpImg = ffmpegToNpImage(rEquiImg, 2880, 2880*2)

    # Define the region of overlap between the images
    h,w = lNpImg.shape[:2]

    if left_color_overlap:
        # left overlapping region
        overlap_region = np.s_[0:h, int(0.225*w):int(0.275*w)] # You'll need to fill this in based on your specific images
    else:
        # right overlapping region
        overlap_region = np.s_[0:h, int(0.725*w):int(0.775*w)] # You'll need to fill this in based on your specific images

    # Extract the overlapping regions from the images
    overlap1 = lNpImg[overlap_region]
    overlap2 = rNpImg[overlap_region]

    # Calculate the color correction parameters
    gain, bias = calculate_correction_params(overlap1, overlap2)
    # print (f"gain {gain} bias {bias}")

    # Apply the color correction parameters to the entire image
    lNpImg = apply_correction_params(lNpImg, gain, bias)


    leveln = 7
    height, width = lNpImg.shape[:2]
    mask = midNpBlendMask(height, width)
    subA = np.zeros(mask.shape)
    subA[:, :lNpImg.shape[1]] = lNpImg
    subB = np.zeros(mask.shape)
    subB[:, -rNpImg.shape[1]:] = rNpImg

    # subAx, subBx, maskx = multi_band_blending.preprocess(lNpImg, rNpImg, 2880, 0)

    # Get Gaussian pyramid and Laplacian pyramid
    MP = multi_band_blending.GaussianPyramid(mask, leveln)
    LPA = multi_band_blending.LaplacianPyramid(subA, leveln)
    LPB = multi_band_blending.LaplacianPyramid(subB, leveln)

    # Blend two Laplacian pyramidspass
    blended = multi_band_blending.blend_pyramid(LPA, LPB, MP)

    # Reconstruction process
    result = multi_band_blending.reconstruct(blended)
    result_min = np.min(result)
    result_max = np.max(result)
    # result = result / leveln
    print (f"result min {result_min} max {result_max}")
    result[result > 255] = 255
    result[result < 0] = 0

    result = (result).astype('uint8')

    if 0:
        result = gray_world(result)
        result = white_patch(result)

    saveNumpyImage(result, buildFilepath("-merged"))


except Exception as e:
    import traceback
    print(traceback.format_exc())
    # print(f"An error occurred: {e}")
    console = code.InteractiveConsole(locals=locals())
    console.interact()
