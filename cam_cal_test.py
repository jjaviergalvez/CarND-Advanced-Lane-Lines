import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

 # load the calibratios parameters
with open('calibration_parameters.p', mode='rb') as f:
    data = pickle.load(f)
mtx, dist = data['cameraMatrix'], data['distCoeffs']

images = glob.glob('test_images/*.jpg')

for idx, fname in enumerate(images):
    # Undistortion on an image
    img = cv2.imread(fname)
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    # Save undistortion images
    write_name = 'undst_test'+str(idx)+'.jpg'
    cv2.imwrite('output_images/' + write_name, dst)

    # Visual undistortion comparition
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    f.savefig('output_images/' + write_name)
