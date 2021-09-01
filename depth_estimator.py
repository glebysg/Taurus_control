# https://github.com/gsurma/stereo_depth_estimator/blob/master/stereo_depth_estimator_sgbm.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from sklearn.preprocessing import normalize

DPI=96
DATASET_LEFT = "stereo_rectified_L/"
DATASET_RIGHT = "stereo_rectified_R/"
DATASET_DISPARITIES = "disparities/"
DATASET_COMBINED = "combined/"
# points = cv2.reprojectImageTo3D(disp, Q)
# colors = cv2.cvtColor(imgU1, cv2.COLOR_BGR2RGB)
def write_ply(fn, verts, colors):
    
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    mask = ~np.isnan(verts).any(axis=1)
    verts = verts[mask]
    colors = colors[mask]
    mask = ~np.isinf(verts).any(axis=1)
    verts = verts[mask]
    colors = colors[mask]
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
        
def process_frame(left, right, name):
	kernel_size = 3
	smooth_left = cv2.GaussianBlur(left, (kernel_size,kernel_size), 1.5)
	smooth_right = cv2.GaussianBlur(right, (kernel_size, kernel_size), 1.5)

	window_size = 9    
	left_matcher = cv2.StereoSGBM_create(
	    numDisparities=96,
	    blockSize=7,
	    P1=8*3*window_size**2,
	    P2=32*3*window_size**2,
	    disp12MaxDiff=1,
	    uniquenessRatio=16,
	    speckleRange=2,
	    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
	)

	right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

	wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
	wls_filter.setLambda(80000)
	wls_filter.setSigmaColor(1.2)

	disparity_left = np.int16(left_matcher.compute(smooth_left, smooth_right))
	disparity_right = np.int16(right_matcher.compute(smooth_right, smooth_left) )

	wls_image = wls_filter.filter(disparity_left, smooth_left, None, disparity_right)
	wls_image = cv2.normalize(src=wls_image, dst=wls_image, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
	wls_image = np.uint8(wls_image)

	fig = plt.figure(figsize=(wls_image.shape[1]/DPI, wls_image.shape[0]/DPI), dpi=DPI, frameon=False);
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	plt.imshow(wls_image, cmap='jet');
	plt.savefig(DATASET_DISPARITIES+name)
	plt.close()
	create_combined_output(left, right, name)

def create_combined_output(left, right, name):
	combined = np.concatenate((left, right, cv2.imread(DATASET_DISPARITIES+name)), axis=0)
	cv2.imwrite(DATASET_COMBINED+name, combined)

def process_dataset():
	left_images = [f for f in os.listdir(DATASET_LEFT) if not f.startswith('.')]
	right_images = [f for f in os.listdir(DATASET_RIGHT) if not f.startswith('.')]
	assert(len(left_images)==len(right_images))
	left_images.sort()
	right_images.sort()
	for i in range(len(left_images)):
		left_image_path = DATASET_LEFT+left_images[i]
		right_image_path = DATASET_RIGHT+right_images[i]
		left_image = cv2.imread(left_image_path, cv2.IMREAD_COLOR)
		right_image = cv2.imread(right_image_path, cv2.IMREAD_COLOR)
		process_frame(left_image, right_image, left_images[i])
		print(i)
if __name__== "__main__":
	process_dataset()