from sklearn.decomposition import FastICA
import matplotlib.image as mpimg
import cv2 
import numpy as np
from cv2 import dnn_superres
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import rasterio
from skimage import io
import tensorflow as tf
import tensorflow_hub as hub
import os
import time
from PIL import Image


class SuperResoulation :

    def __init__(self, image_name) : 
        self.image_name = image_name
        self.output = None
        
 
    



class ImageOperations  :

    
    class HistogramStreatching :

        def __init__(self, img_path, output_path) : 
            self.img_path = img_path
            self.output_path = output_path

        def Histogram_Equalization(self):
        # Read the image
            img = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)

            # Perform histogram equalization
            equ = cv2.equalizeHist(img)

            # Save the equalized image
            cv2.imwrite(self.output_path, equ)

        def Minimum_Maximum (self) : 
            # Load the input image
            img = cv2.imread(self.img_path, cv2.IMREAD_COLOR)

            # Split the image into color channels
            r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

            # Plot the histograms for each channel
            hist_r = np.zeros(256)
            hist_g = np.zeros(256)
            hist_b = np.zeros(256)

            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    hist_r[r[i,j]] += 1
                    hist_g[g[i,j]] += 1
                    hist_b[b[i,j]] += 1

            plt.plot(hist_r, color='red', alpha=0.10)
            plt.plot(hist_g, color='green', alpha=0.10)
            plt.plot(hist_b, color='blue', alpha=0.10)

            # Stretch the contrast for each channel
            min_r, max_r = np.min(r), np.max(r)
            min_g, max_g = np.min(g), np.max(g)
            min_b, max_b = np.min(b), np.max(b)

            re_stretch = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            gr_stretch = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            bl_stretch = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    re_stretch[i,j] = int((r[i,j] - min_r) * 255 / (max_r - min_r))
                    gr_stretch[i,j] = int((g[i,j] - min_g) * 255 / (max_g - min_g))
                    bl_stretch[i,j] = int((b[i,j] - min_b) * 255 / (max_b - min_b))

            # Merge the channels back together
            img_stretch = cv2.merge((re_stretch, gr_stretch, bl_stretch))

            cv2.imwrite(self.output_path, img_stretch)
        
        def Standard_Deviation (self) : 
            
            image = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)

            # Define the standard deviation for stretching
            std_dev = 2.0  # You can adjust this value

            # Calculate the mean and standard deviation of the image
            mean, std = cv2.meanStdDev(image)

            # Calculate lower and upper bounds based on standard deviation
            lower_bound = int(mean - std_dev * std)
            upper_bound = int(mean + std_dev * std)

            # Clip pixel values to ensure they fall within the bounds
            stretched_image = np.clip(image, lower_bound, upper_bound)

            # Normalize the stretched image to the full 0-255 range
            stretched_image = cv2.normalize(stretched_image, None, 0, 255, cv2.NORM_MINMAX)

            # Convert to uint8 (8-bit) data type
            stretched_image = stretched_image.astype(np.uint8)


            # Save the stretched image if needed
            cv2.imwrite(self.output_path, stretched_image)



    class ImageFilter :

        def __init__(self, img_path, output_path) -> None:
            self.img_path = img_path
            self.output_path = output_path

        def Sobel_Filter(self):
            # Load the image
            img0 = cv2.imread(self.img_path)

            # Resize the image to a smaller size
            new_height, new_width = 800, 600  # Adjust the dimensions as needed
            img0 = cv2.resize(img0, (new_width, new_height))

            # Convert to grayscale
            gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

            # Remove noise using Gaussian blur
            img = cv2.GaussianBlur(gray, (3, 3), 0)

            # Convolute with proper kernels
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # Sobel Y

            # Save the Sobel Y image
            cv2.imwrite(self.output_path, sobely)        


            
        def High_Pass_Filter (self) : 
            # Load the image
            image = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)

            # Create a 5x5 kernel for the high-pass filter
            kernel = np.array([[-1, -1, -1, -1, -1],
                            [-1,  1,  2,  1, -1],
                            [-1,  2,  4,  2, -1],
                            [-1,  1,  2,  1, -1],
                            [-1, -1, -1, -1, -1]])

            # Apply the filter using convolution
            high_pass_image = cv2.filter2D(image, -1, kernel)

            # Ensure pixel values are in the valid range (0-255)
            high_pass_image = np.clip(high_pass_image, 0, 255)

            # Save the filtered image
            cv2.imwrite(self.output_path, high_pass_image)


        def Low_Pass_Filter (self) :
            # Load the image
            image = cv2.imread(self.img_path)

            # Define the size and standard deviation of the Gaussian kernel
            kernel_size = (5, 5)
            sigma = 1.0

            # Create the Gaussian kernel
            gaussian_kernel = cv2.getGaussianKernel(kernel_size[0], sigma)
            gaussian_kernel = gaussian_kernel * gaussian_kernel.T

            # Apply the filter using convolution
            low_pass_image = cv2.filter2D(image, -1, gaussian_kernel)

            # Save the filtered image
            cv2.imwrite(self.output_path, low_pass_image)


        def Remove_Noise (self) : 
            # Load the image
            image = cv2.imread(self.img_path)

            #Plot the original image
            plt.subplot(1, 2, 1)
            plt.title("Original")
            plt.imshow(image)

            # Remove noise using a median filter
            filtered_image = cv2.medianBlur(image, 11)

            cv2.imwrite(self.output_path, filtered_image)


    class ImageTransformattion :
        
        def __init__(self, img_path, output_path) : 
            self.img_path = img_path
            self.output_path = output_path

        def PCA (self) : 
            img = cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)

            blue, green, red = cv2.split(img)

            df_blue = blue / 255
            df_green = green / 255
            df_red = red / 255

            pca_b = PCA(n_components=50)
            pca_b.fit(df_blue)
            trans_pca_b = pca_b.transform(df_blue)

            pca_g = PCA(n_components=50)
            pca_g.fit(df_green)
            trans_pca_g = pca_g.transform(df_green)

            pca_r = PCA(n_components=50)
            pca_r.fit(df_red)
            trans_pca_r = pca_r.transform(df_red)

            # Scale back to the original range (0-255)
            b_arr = np.clip(pca_b.inverse_transform(trans_pca_b) * 255, 0, 255).astype(np.uint8)
            g_arr = np.clip(pca_g.inverse_transform(trans_pca_g) * 255, 0, 255).astype(np.uint8)
            r_arr = np.clip(pca_r.inverse_transform(trans_pca_r) * 255, 0, 255).astype(np.uint8)

            img_reduced = cv2.merge((b_arr, g_arr, r_arr))

            # Save the compressed image as a TIFF file
            cv2.imwrite(self.output_path, img_reduced, [cv2.IMWRITE_TIFF_COMPRESSION, 1])  # Set TIFF compression (1 is for no compression, you can adjust this)

        def NDVI (self) : 
            # Load an RGB image
            image = io.imread(self.img_path)  # Replace with the path to your image

            # Assuming the image is in the form of (height, width, channels)
            if len(image.shape) == 3 and image.shape[2] >= 3:  # Ensure it's a color image with 3 channels
                # Extract individual bands (assuming the image is in RGB format)
                red = image[:, :, 0].astype(np.float32)
                nir = image[:, :, 2].astype(np.float32)

                # Calculate NDVI
                ndvi = (nir - red) / (nir + red + 1e-8)  # Adding a small value to prevent division by zero

                # Scale NDVI to 0-255 and convert to uint8 for image saving
                ndvi_scaled = ((ndvi + 1) * 127.5).astype(np.uint8)

                # Save NDVI as an image file (adjust the file format if needed)
                output_file = self.output_path
                io.imsave(output_file, ndvi_scaled)

        def perform_ica(self,hyper_image, n_components):
            # Reshape the hyper-image into a 2D array (pixels as rows, bands as columns)
            num_pixels = hyper_image.shape[0] * hyper_image.shape[1]
            num_bands = hyper_image.shape[2]
            flattened_image = hyper_image.reshape(num_pixels, num_bands)

            # Perform ICA
            ica = FastICA(n_components=n_components, random_state=0)
            ica_result = ica.fit_transform(flattened_image)

            # Reshape ICA components back to the original image shape
            ica_image = ica_result.reshape(hyper_image.shape)

            return ica_image
        
        def ICA (self) : 

            image = plt.imread(self.img_path)

            # Reshape the input image into a 2D array
            num_pixels = image.shape[0] * image.shape[1]
            num_bands = image.shape[2]
            flattened_image = image.reshape(num_pixels, num_bands)

            # Perform ICA
            ica = FastICA(n_components=3, random_state=0)
            ica_result = ica.fit_transform(flattened_image)

            # Reshape ICA components back to the original image shape
            ica_image = ica_result.reshape(image.shape)

            # Save the transformed image in the same directory as the script
            output_image_path = self.output_path
            Image.fromarray((ica_image * 255).astype(np.uint8)).save(output_image_path)


        def mnf_transform(self, image_path, num_components):
            # Load the image (you might use a library like OpenCV or scikit-image for this)
            image = plt.imread(image_path)

            # Reshape the input image into a 2D array
            num_pixels = image.shape[0] * image.shape[1]
            num_bands = image.shape[2]
            flattened_image = image.reshape(num_pixels, num_bands)

            # Compute the covariance matrix
            covariance_matrix = np.cov(flattened_image, rowvar=False)

            # Compute eigenvalues and eigenvectors of the covariance matrix
            eigenvalues, eigenvectors = eigh(covariance_matrix)

            # Sort eigenvalues and eigenvectors in descending order
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]

            # Select the top 'num_components' eigenvectors
            selected_eigenvectors = eigenvectors[:, :num_components]

            # Transform the data to MNF space
            mnf_result = np.dot(flattened_image, selected_eigenvectors)

            # Reshape MNF components back to the original image shape
            mnf_image = mnf_result.reshape(image.shape)


            # Save the transformed image in the same directory as the script
            output_image_path = self.output_path
            Image.fromarray((mnf_image * 255).astype(np.uint8)).save(output_image_path)

            return output_image_path
        
        def MNF(self):
            image_path = self.img_path
            num_components = 3
            transformed_image_path = self.mnf_transform(image_path, num_components)
            return transformed_image_path

    class ImageResolution :
        def __init__(self, img_path, output_path) : 
            self.img_path = img_path
            self.output_path = output_path

        def EDSR (self) : 
            # Create an SR object
            sr = dnn_superres.DnnSuperResImpl_create()

            # Read image
            image = cv2.imread(self.img_path)

            # Read the desired model
            path = "ajax_app/Models/EDSR_x4.pb"
            sr.readModel(path)

            # Set the desired model and scale to get correct pre- and post-processing
            sr.setModel("edsr", 4)

            # Upscale the image
            result = sr.upsample(image)

            # Save the image
            cv2.imwrite(self.output_path, result)
        
        def ESRGAN (self) : 
            hr_image = self.preprocess_image(self.img_path)

            # Plotting Original Resolution image
            self.plot_image(tf.squeeze(hr_image), title=self.output_path)
            self.save_image(tf.squeeze(hr_image), filename=self.output_path)

        def plot_image(self,image, title=""):
            """
                Plots images from image tensors.
                Args:
                image: 3D image tensor. [height, width, channels].
                title: Title to display in the plot.
            """
            image = np.asarray(image)
            image = tf.clip_by_value(image, 0, 255)
            image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
            plt.imshow(image)
            plt.axis("off")
            plt.title(title)
            plt.close()

        def save_image(self,image, filename):
            """
                Saves unscaled Tensor Images.
                Args:
                image: 3D image tensor. [height, width, channels]
                filename: Name of the file to save.
            """
            if not isinstance(image, Image.Image):
                image = tf.clip_by_value(image, 0, 255)
                image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
            image.save(filename)

        def preprocess_image(self,image_path):

                """ Loads image from path and preprocesses to make it model ready
                    Args:
                        image_path: Path to the image file
                """
                hr_image = tf.image.decode_image(tf.io.read_file(image_path))
                # If PNG, remove the alpha channel. The model only supports
                # images with 3 color channels.
                if hr_image.shape[-1] == 4:
                    hr_image = hr_image[...,:-1]
                hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
                hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
                hr_image = tf.cast(hr_image, tf.float32)
                return tf.expand_dims(hr_image, 0)
