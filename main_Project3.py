import numpy as np

####################################################################################################
# Exercise 1: DFT

def dft_matrix(n: int) -> np.ndarray:
    """
    Construct DFT matrix of size n.

    Arguments:
    n: size of DFT matrix

    Return:
    F: DFT matrix of size n

    """
    F = np.zeros((n, n), dtype='complex128')
    
    w = np.zeros(n, dtype='complex128')
    w = np.exp(((-2*np.pi) * 1j)/ n)
        
    for i in range (n):
        for i_2 in range (n):
            F[i][i_2] = np.power(w,(i_2 * i))
    F = np.dot((1/np.sqrt(n)), F)
    

    return F


def is_unitary(matrix: np.ndarray) -> bool:
    """
    Check if the passed in matrix of size (n times n) is unitary.

    Arguments:
    matrix: the matrix which is checked

    Return:
    unitary: True if the matrix is unitary
    """
    
    mcon= matrix.conjugate().T
    return np.allclose(np.dot(mcon, matrix), np.identity(len(matrix)))


def create_harmonics(n: int = 128) -> (list, list):
    """
    Create delta impulse signals and perform the fourier transform on each signal.

    Arguments:
    n: the length of each signal

    Return:
    sigs: list of np.ndarrays that store the delta impulse signals
    fsigs: list of np.ndarrays with the fourier transforms of the signals
    """
    sigs = []
    sigs_m = np.identity(n)
    for i in range (n):
        sigs.append(sigs_m[i])
    
    fsigs = []
    for i in range (n):
        fsigs.append(np.dot(dft_matrix(n), sigs[i]))


    return sigs, fsigs


####################################################################################################
# Exercise 2: FFT


def shuffle_bit_reversed_order(data: np.ndarray) -> np.ndarray:
    """
    Shuffle elements of data using bit reversal of list index.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    data: shuffled data array
    """
    datacopy = data.copy()
    data_b = []
    data_b_r = []
    data_i = np.zeros(len(data), dtype = int)
    width = len("{0:b}".format(len(data)-1))
    loadi = 0
    for i in range (len(data)):
        data_b.append('{:0{width}b}'.format(i, width=width))
    for b in data_b:
       data_b_r.append(b[::-1])
    for b in data_b_r:
        data_i[loadi] = int(b, 2)
        loadi += 1
    for i in range (len(data_i)):
        data[i] = datacopy[data_i[i]]   
    return data


def fft(data: np.ndarray) -> np.ndarray:
    """
    Perform real-valued discrete Fourier transform of data using fast Fourier transform.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    fdata: Fourier transformed data

    Note:
    This is not an optimized implementation but one to demonstrate the essential ideas
    of the fast Fourier transform.

    """

    fdata = np.asarray(data, dtype='complex128')
    n = fdata.size
    fdata = shuffle_bit_reversed_order(fdata)
    if not n > 0 or (n & (n - 1)) != 0:
        raise ValueError

    

    treedepth = int(np.log2(n))
    for m in range (treedepth):
        for k in range (2**m):
            for i in range (0, n, (2**(m+1))):
                i = i+k
                j = i + 2**m 
                omega = np.exp((-2 * np.pi * 1j* k)/(2**(m+1))) * fdata[j]
                fdata[j] = fdata[i] - omega
                fdata[i] = fdata[i] + omega
            
            
    fdata = (1/np.sqrt(n))* fdata           
    return fdata


def generate_tone(f: float = 261.626, num_samples: int = 44100) -> np.ndarray:
    """
    Generate tone of length 1s with frequency f (default mid C: f = 261.626 Hz) and return the signal.

    Arguments:
    f: frequency of the tone

    Return:
    data: the generated signal
    """

    # sampling range
    x_min = 0.0
    x_max = 1.0
    data = np.zeros(num_samples)

    for i in range (num_samples):
        data[i] = np.sin(2*np.pi*f*(i/(num_samples-1)))

    return data


def low_pass_filter(adata: np.ndarray, bandlimit: int = 1000, sampling_rate: int = 44100) -> np.ndarray:
    """
    Filter high frequencies above bandlimit.

    Arguments:
    adata: data to be filtered
    bandlimit: bandlimit in Hz above which to cut off frequencies
    sampling_rate: sampling rate in samples/second

    Return:
    adata_filtered: filtered data
    """
    
    # translate bandlimit from Hz to dataindex according to sampling rate and data size
    bandlimit_index = int(bandlimit*adata.size/sampling_rate)

    fdata = np.fft.fft(adata)
    for i in range(bandlimit_index + 1, adata.size - bandlimit_index):
            fdata[i] = 0
    

    adata_filtered = np.fft.ifft(fdata)
    adata_filtered = np.real(adata_filtered)
    return adata_filtered


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
