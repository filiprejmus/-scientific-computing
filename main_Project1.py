import numpy as np
import lib
import matplotlib as mpl


####################################################################################################

def power_iteration(M: np.ndarray, epsilon: float = -1.0) -> (np.ndarray, list):
    """
    Compute largest eigenvector of matrix M using power iteration. It is assumed that the
    largest eigenvalue of M, in magnitude, is well separated.

    Arguments:
    M: matrix, assumed to have a well separated largest eigenvalue
    epsilon: epsilon used for convergence (default: 10 * machine precision)

    Return:
    vector: eigenvector associated with largest eigenvalue
    residuals : residual for each iteration step

    Raised Exceptions:
    ValueError: if matrix is not square
    """

    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix not nxn")

    
    epsilon = np.finfo(np.float64).eps * 10

    # random vector of proper size to initialize iteration
    vector = np.random.rand(M.shape[1])
    

    # Initialize residual list and residual of current eigenvector estimate
    residuals = []
    
    vector = np.random.rand(M.shape[1])
    residuals = []
    residual = 2 * epsilon
    while residual > epsilon:
        newvector = M.dot(vector)
        newvector /= np.linalg.norm(newvector)
        residual = np.linalg.norm(newvector - vector)
        residuals.append(residual)
        vector = newvector
 
    return vector, residuals


####################################################################################################
# Exercise 2: Eigenfaces

def load_images(path: str, file_ending: str=".png") -> (list, int, int):
    """
    Load all images in path with matplotlib that have given file_ending

    Arguments:
    path: path of directory containing image files that can be assumed to have all the same dimensions
    file_ending: string that image files have to end with, if not->ignore file

    Return:
    images: list of images (each image as numpy.ndarray and dtype=float64)
    dimension_x: size of images in x direction
    dimension_y: size of images in y direction
    """

    images = []

    # read each image in path as numpy.ndarray and append to images
    # Useful functions: lib.list_directory(), matplotlib.image.imread(), numpy.asarray()


    #set dimensions according to first image in images
    imglist = lib.list_directory(path)
    imglist.sort()
    for x in imglist:
        if x.endswith((file_ending)):
            images.append(np.asarray(mpl.image.imread(path+x), dtype=np.float64))
        
        
  
    dimension_y = images[1].shape[0]
    dimension_x = images[1].shape[1]
    
    

    return images, dimension_x, dimension_y


def setup_data_matrix(images: list) -> np.ndarray:
    """
    Create data matrix out of list of 2D data sets.

    Arguments:
    images: list of 2D images (assumed to be all homogeneous of the same size and type np.ndarray)

    Return:
    D: data matrix that contains the flattened images as rows
    """

    D = np.zeros((len(images), images[0].shape[0]*images[0].shape[1]))
    for imgcnt in range (len(images)):
        D[imgcnt] = images[imgcnt].flatten()


    return D


def calculate_pca(D: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Perform principal component analysis for given data matrix.

    Arguments:
    D: data matrix of size m x n where m is the number of observations and n the number of variables

    Return:
    pcs: matrix containing principal components as rows
    svals: singular values associated with principle components
    mean_data: mean that was subtracted from data
    """
    m = D.shape[0]
    mean_data = np.mean(D, axis=0, dtype = np.float64)
    
    for i in range (m):
        D[i] = D[i] - mean_data    
            
    u, svals, v = np.linalg.svd(D, full_matrices=False)
    pcs = v
    return pcs, svals, mean_data


def accumulated_energy(singular_values: np.ndarray, threshold: float = 0.8) -> int:
    """
    Compute index k so that threshold percent of magnitude of singular values is contained in
    first k singular vectors.

    Arguments:
    singular_values: vector containing singular values
    threshold: threshold for determining k (default = 0.8)

    Return:
    k: threshold index
    """
    k = 0
    sin_sum = np.sum(singular_values)
    for x in range (singular_values.shape[0]):
        singular_values[x] /= sin_sum
        
    sin_search = 0.0
    while sin_search < threshold:
        sin_search += singular_values[k]
        k += 1
    
    return k


def project_faces(pcs: np.ndarray, images: list, mean_data: np.ndarray) -> np.ndarray:
    """
    Project given image set into basis.

    Arguments:
    pcs: matrix containing principal components / eigenfunctions as rows
    images: original input images from which pcs were created
    mean_data: mean data that was subtracted before computation of SVD/PCA

    Return:
    coefficients: basis function coefficients for input images, each row contains coefficients of one image
    """
    
    D = setup_data_matrix(images)
    for i in range (D.shape[0]):
        D[i] -= mean_data
    coefficients = D.dot(pcs.T)



    return coefficients


def identify_faces(coeffs_train: np.ndarray, pcs: np.ndarray, mean_data: np.ndarray, path_test: str) -> (
np.ndarray, list, np.ndarray):
    """
    Perform face recognition for test images assumed to contain faces.

    For each image coefficients in the test data set the closest match in the training data set is calculated.
    The distance between images is given by the angle between their coefficient vectors.

    Arguments:
    coeffs_train: coefficients for training images, each image is represented in a row
    path_test: path to test image data

    Return:
    scores: Matrix with correlation between all train and test images, train images in rows, test images in columns
    imgs_test: list of test images
    coeffs_test: Eigenface coefficient of test images
    """

    imgs_test, x, y = load_images(path_test)
    coeffs_test = project_faces(pcs, imgs_test, mean_data)
    train_rows = coeffs_train.shape[0]
    test_rows = coeffs_test.shape[0]
    scores = np.zeros((train_rows, test_rows))
    for i in range (train_rows):
        for j in range (test_rows):
            scores[i][j] = np.arccos(np.dot(coeffs_train[i],coeffs_test[j])/(np.linalg.norm(coeffs_train[i])*np.linalg.norm(coeffs_test[j])))


    return scores, imgs_test, coeffs_test


if __name__ == '__main__':

    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
