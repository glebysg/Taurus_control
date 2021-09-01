# https://github.com/gsurma/stereo_depth_estimator/blob/master/stereo_depth_estimator_sgbm.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from sklearn.preprocessing import normalize

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
 
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
	# left matcher is StereoSGBM_create
	# left_matcher = cv2.StereoSGBM_create(
	#     numDisparities=96,
	#     blockSize=7,
	#     P1=8*3*window_size**2,
	#     P2=32*3*window_size**2,
	#     disp12MaxDiff=1,
	#     uniquenessRatio=16,
	#     speckleRange=2,
	#     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
	# )
	left_matcher = cv2.StereoSGBM_create(
     	# minDisparity=0,
      	preFilterCap=63,
       	# speckleWindowSize=0,
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
	# Convenience method to set up the matcher for computing the right-view disparity map that is required in case of filtering with confidence.
	# Disparity map filter based on Weighted Least Squares filter
	wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
	wls_filter.setLambda(80000) # changing this not very effective.
	wls_filter.setSigmaColor(1.2)

	disparity_left = np.int16(left_matcher.compute(smooth_left, smooth_right))
	disparity_right = np.int16(right_matcher.compute(smooth_right, smooth_left) )
	# https://docs.opencv.org/3.4/db/d72/classcv_1_1ximgproc_1_1DisparityFilter.html
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
	##################################save point cloud
	return 
	h, w = wls_image.shape[:2]
	f = 0.8*w                          # guess for focal length
	Q = np.float32([[1, 0, 0, -0.5*w],
					[0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
					[0, 0, 0,     -f], # so that y-axis looks up
					[0, 0, 1,      0]])
	points = cv2.reprojectImageTo3D(wls_image, Q)
	colors = cv2.cvtColor(smooth_left, cv2.COLOR_BGR2RGB)
	mask = wls_image > wls_image.min()
	out_points = points[mask]
	out_colors = colors[mask]
	write_ply(f"ply/{name}.ply",points, colors)
 
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
		break
if __name__== "__main__":
	process_dataset()