import imageio
import os

def get_gif(plot_dir):
	filenames = os.listdir(plot_dir)
	filenames = filenames[:-1]
	print(filenames)
	images = []
	for filename in filenames:
		images.append(imageio.imread(os.path.join(plot_dir, filename)))
	imageio.mimsave(os.path.join(plot_dir, "viz.gif"), images, duration=2)