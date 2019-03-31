import numpy as np


####################################################################################################
# Exercise 1: Interpolation

def lagrange_interpolation(x: np.ndarray, y: np.ndarray) -> (np.poly1d, list):
    """
    Generate Lagrange interpolation polynomial.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    polynomial: polynomial as np.poly1d object
    base_functions: list of base polynomials
    """

    assert (x.size == y.size)
    polynomial = np.poly1d(0)
    base_functions = []
    for i in range (x.size):
        hilfsarray = []
        for j in range (x.size):
            if i != j:
                hilfsarray.append(x[j])
        base_functionpoly = np.poly1d(hilfsarray, True)
        zaehler = base_functionpoly(x[i])
        base_functions.append(base_functionpoly/zaehler)
        polynomial += base_functions[i]*y[i]
        
        
    return polynomial, base_functions




def hermite_cubic_interpolation(x: np.ndarray, y: np.ndarray, yp: np.ndarray) -> list:
    """
    Compute hermite cubic interpolation spline

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points
    yp: derivative values of interpolation points

    Returns:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size == yp.size)

    spline = []
    spline_matrix = np.zeros(shape = (4,4))
    for i in range (x.size - 1):
        spline_matrix[0] = ([1, x[i], x[i]**2, x[i]**3])
        spline_matrix[1] = ([1, x[i+1], x[i+1]**2, x[i+1]**3])
        spline_matrix[2] = ([0, 1, 2*x[i], 3*x[i]**2])
        spline_matrix[3] = ([0, 1, 2*x[i+1], 3*x[i+1]**2])
        spline_matrix = np.linalg.inv(spline_matrix)
        poly = (np.poly1d([spline_matrix[3][0], spline_matrix[2][0], spline_matrix[1][0], spline_matrix[0][0]])*y[i])+(np.poly1d([spline_matrix[3][1], spline_matrix[2][1], spline_matrix[1][1], spline_matrix[0][1]]) * y[i+1]) + (np.poly1d([spline_matrix[3][2] ,spline_matrix[2][2], spline_matrix[1][2], spline_matrix[0][2]]) * yp[i]) + (np.poly1d([spline_matrix[3][3],spline_matrix[2][3],spline_matrix[1][3],spline_matrix[0][3]]) * yp[i + 1])
        spline.append(poly)
    return spline



####################################################################################################
# Exercise 2: Animation

def natural_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Intepolate the given function using a spline with natural boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    m_len = x.size
    spline_matrix = np.zeros(shape = (4*(m_len-1), 4*(m_len-1)))
    for i in range (m_len-1):
            j = i
            i *= 4
            spline_matrix[i][i] = 1
            spline_matrix[i][i+1] = x[j]
            spline_matrix[i][i+2] = x[j] ** 2
            spline_matrix[i][i+3] = x[j] ** 3
            spline_matrix[i+1][i] = 1
            spline_matrix[i+1][i+1] = x[j+1]
            spline_matrix[i+1][i+2] = x[j+1] ** 2
            spline_matrix[i+1][i+3] = x[j+1] ** 3
            if j == (m_len - 2):
                break
            spline_matrix[i+2][i] = 0
            spline_matrix[i+2][i+1] = 1
            spline_matrix[i+2][i+2] = 2 * x[j+1]
            spline_matrix[i+2][i+3] = 3 * x[j+1] ** 2
            spline_matrix[i+3][i] = 0
            spline_matrix[i+3][i+1] = 0
            spline_matrix[i+3][i+2] = 2 
            spline_matrix[i+3][i+3] = 6 * x[j+1]
            spline_matrix[i+2][i+4] = 0
            spline_matrix[i+2][i+5] = -1
            spline_matrix[i+2][i+6] = -2 * x[j+1]
            spline_matrix[i+2][i+7] = -3 * x[j+1] ** 2
            spline_matrix[i+3][i+4] = 0
            spline_matrix[i+3][i+5] = 0
            spline_matrix[i+3][i+6] = -2 
            spline_matrix[i+3][i+7] = -6 * x[j+1]
            
    spline_matrix[4*m_len-6][2]= 2
    spline_matrix[4*m_len-6][3]= 6 * x[0]
    spline_matrix[4*m_len-5][4*m_len-6]= 2
    spline_matrix[4*m_len-5][4*m_len-5]= 6 * x[x.size-1]
    solve_vector = np.ndarray(4*(m_len-1))
    for i in range (m_len - 1):
        j = i
        i *= 4
        solve_vector[i] = y[j]
        solve_vector[i+1] = y[j+1]
        solve_vector[i+2] = 0
        solve_vector[i+3] = 0
    spline_array = np.linalg.solve(spline_matrix, solve_vector)
    spline= []
    for i in range (m_len - 1):
        spline.append(np.poly1d([spline_array[4*i+3],spline_array[4*i+2],spline_array[4*i+1],spline_array[4*i]]))
    


    return spline


def periodic_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Interpolate the given function with a cubic spline and periodic boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    m_len = x.size
    spline_matrix = np.zeros(shape = (4*(m_len-1), 4*(m_len-1)))
    for i in range (m_len-1):
            j = i
            i *= 4
            spline_matrix[i][i+3] = 1
            spline_matrix[i][i+2] = x[j]
            spline_matrix[i][i+1] = x[j] ** 2
            spline_matrix[i][i] = x[j] ** 3
            spline_matrix[i+1][i+3] = 1
            spline_matrix[i+1][i+2] = x[j+1]
            spline_matrix[i+1][i+1] = x[j+1] ** 2
            spline_matrix[i+1][i] = x[j+1] ** 3
            if j == (m_len - 2):
                break
            spline_matrix[i+2][i+3] = 0
            spline_matrix[i+2][i+2] = 1
            spline_matrix[i+2][i+1] = 2 * x[j+1]
            spline_matrix[i+2][i] = 3 * x[j+1] ** 2
            spline_matrix[i+3][i+3] = 0
            spline_matrix[i+3][i+2] = 0
            spline_matrix[i+3][i+1] = 2 
            spline_matrix[i+3][i] = 6 * x[j+1]
            spline_matrix[i+2][i+7] = 0
            spline_matrix[i+2][i+6] = -1
            spline_matrix[i+2][i+5] = -2 * x[j+1]
            spline_matrix[i+2][i+4] = -3 * x[j+1] ** 2
            spline_matrix[i+3][i+7] = 0
            spline_matrix[i+3][i+6] = 0
            spline_matrix[i+3][i+5] = -2 
            spline_matrix[i+3][i+4] = -6 * x[j+1]
            
            
    spline_matrix[4*m_len-6][2]= 1      
    spline_matrix[4*m_len-6][1]= 2 * x[0]
    spline_matrix[4*m_len-6][0]= 3 * x[0] ** 2
    spline_matrix[4*m_len-5][1]= 2
    spline_matrix[4*m_len-5][0]= 6 * x[0]
    spline_matrix[4*m_len-6][4*m_len-6]= -1      
    spline_matrix[4*m_len-6][4*m_len-7]= -2 * x[m_len-1]
    spline_matrix[4*m_len-6][4*m_len-8]= -3 * x[m_len-1] ** 2
    spline_matrix[4*m_len-5][4*m_len-7]= -2
    spline_matrix[4*m_len-5][4*m_len-8]= -6 * x[x.size-1]
    solve_vector = np.ndarray(4*(m_len-1))
    for i in range (m_len - 1):
        j = i
        i *= 4
        solve_vector[i] = y[j]
        solve_vector[i+1] = y[j+1]
        solve_vector[i+2] = 0
        solve_vector[i+3] = 0
    spline_array = np.linalg.solve(spline_matrix, solve_vector)
    spline= []
    for i in range (m_len - 1):
        spline.append(np.poly1d([spline_array[4*i],spline_array[4*i+1],spline_array[4*i+2],spline_array[4*i+3]]))
    

    return spline


if __name__ == '__main__':

    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
