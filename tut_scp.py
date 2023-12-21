# from scipy.misc import imread, imsave, imresize
# from imageio import imread, imsave

from imageio.v2 import imread, imsave

img = imread("kitty-cat.jpeg")

img_tint = img * [1, 0.45, 0.3]

imsave("kitty_tint.png", img_tint)

# img_tint_resize = imresize(img_tint, (300, 300))

# imsave("kitty_tint_resized.png", img_tint_resize)

print(img.dtype, img.shape)