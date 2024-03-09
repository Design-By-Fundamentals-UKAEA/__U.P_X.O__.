'''
https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/

https://pypi.org/project/poissonDiskSampling/

bridson1 offers consant density Poisson disk sampling
IMPLEMENTED
    https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/Bridson_sampling.py

bridson2 also offers consant density Poisson disk sampling
TO BE IMPLEMENTED
    https://github.com/diregoblin/poisson_disc_sampling

bridson3 also offers consant density Poisson disk sampling
TO BE IMPLEMENTED
https://github.com/emulbreh/bridson

bridson4 offers variable density Poisson disk sampling
TO BE IMPLEMENTED
https://pypi.org/project/poissonDiskSampling/

dart1 offers constant density dart sampling
    https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/DART_sampling_numpy.py
'''
###############################################################################
import numpy as np
#import matplotlib.pyplot as plt
def bridson1(width = 1.0, height = 1.0, radius = 0.01, k = 10):
    # Credit: https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/Bridson_sampling.py
    # -----------------------------------------------------------------------------
    # From Numpy to Python
    # Copyright (2017) Nicolas P. Rougier - BSD license
    # More information at https://github.com/rougier/numpy-book
    # -----------------------------------------------------------------------------
    # References: Fast Poisson Disk Sampling in Arbitrary Dimensions
    #             Robert Bridson, SIGGRAPH, 2007
    def squared_distance(p0, p1):
        return (p0[0]-p1[0])**2 + (p0[1]-p1[1])**2
    def random_point_around(p, k=1):
        # WARNING: This is not uniform around p but we can live with it
        R = np.random.uniform(radius, 2*radius, k)
        T = np.random.uniform(0, 2*np.pi, k)
        P = np.empty((k, 2))
        P[:, 0] = p[0]+R*np.sin(T)
        P[:, 1] = p[1]+R*np.cos(T)
        return P
    def in_limits(p):
        return 0 <= p[0] < width and 0 <= p[1] < height
    def neighborhood(shape, index, n=2):
        row, col = index
        row0, row1 = max(row-n, 0), min(row+n+1, shape[0])
        col0, col1 = max(col-n, 0), min(col+n+1, shape[1])
        I = np.dstack(np.mgrid[row0:row1, col0:col1])
        I = I.reshape(I.size//2, 2).tolist()
        I.remove([row, col])
        return I
    def in_neighborhood(p):
        i, j = int(p[0]/cellsize), int(p[1]/cellsize)
        if M[i, j]:
            return True
        for (i, j) in N[(i, j)]:
            if M[i, j] and squared_distance(p, P[i, j]) < squared_radius:
                return True
        return False
    def add_point(p):
        points.append(p)
        i, j = int(p[0]/cellsize), int(p[1]/cellsize)
        P[i, j], M[i, j] = p, True
    # Here `2` corresponds to the number of dimension
    cellsize = radius/np.sqrt(2)
    rows = int(np.ceil(width/cellsize))
    cols = int(np.ceil(height/cellsize))
    # Squared radius because we'll compare squared distance
    squared_radius = radius*radius
    # Positions cells
    P = np.zeros((rows, cols, 2), dtype=np.float32)
    M = np.zeros((rows, cols), dtype=bool)
    # Cache generation for neighborhood
    N = {}
    for i in range(rows):
        for j in range(cols):
            N[(i, j)] = neighborhood(M.shape, (i, j), 2)
    points = []
    add_point((np.random.uniform(width), np.random.uniform(height)))
    while len(points):
        i = np.random.randint(len(points))
        p = points[i]
        del points[i]
        Q = random_point_around(p, k)
        for q in Q:
            if in_limits(q) and not in_neighborhood(q):
                add_point(q)
    return P[M]
# =============================================================================
#     if __name__ == '__main__':
#         plt.figure()
#         plt.subplot(1, 1, 1, aspect=1)
#         points = Bridson_sampling()
#         X = [x for (x, y) in points]
#         Y = [y for (x, y) in points]
#         plt.scatter(X, Y, s=2)
#         plt.xlim(0, 1)
#         plt.ylim(0, 1)
#         plt.show()
# =============================================================================
###############################################################################
def bridson_variable_density():
    # https://pypi.org/project/poissonDiskSampling/
    pass
###############################################################################
def dart(width = 1.0, height = 1.0, radius = 0.025, k = 30):
    # -----------------------------------------------------------------------------
    # Credit: https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/DART_sampling_numpy.py
    # -----------------------------------------------------------------------------
    # From Numpy to Python
    # Copyright (2017) Nicolas P. Rougier - BSD license
    # More information at https://github.com/rougier/numpy-book
    # -----------------------------------------------------------------------------
    # Return array type modified by Sunil Anandatheertha, Material Engineer, UKAEA
    # Also, some simplications to calculations were introduced by Sunil Anandatheertha
    # -----------------------------------------------------------------------------
    import numpy as np
    #import matplotlib.pyplot as plt
    from scipy.spatial.distance import cdist
    # 5 times the Theoretical limit
    # n = 5*int((width+radius)*(height+radius) / (2*(radius/2)*(radius/2)*np.sqrt(3))) + 1
    # Simplication of the above commented out expression
    n = 5*int((width+radius)*(height+radius) / (0.5*radius*radius*1.73205080757)) + 1
    # Compute n random points
    P = np.zeros((n, 2))
    P[:, 0] = np.random.uniform(0, width, n)
    P[:, 1] = np.random.uniform(0, height, n)
    # TODO: Simplify the above three lines of codes to  single line
    # Computes respective distances at once
    D = cdist(P, P)
    # Cancel null distances on the diagonal
    D[range(n), range(n)] = 1e10
    points, indices = [P[0]], [0]
    i = 1
    last_success = 0
    while i < n and i - last_success < k:
        if D[i, indices].min() > radius:
            indices.append(i)
            points.append(P[i])
            last_success = i
        i += 1
    return [[_[0],_[1]] for _ in points]
# =============================================================================
#     if __name__ == '__main__':
#     
#         plt.figure()
#         plt.subplot(1, 1, 1, aspect=1)
#     
#         points = DART_sampling_numpy()
#         X = [x for (x, y) in points]
#         Y = [y for (x, y) in points]
#         plt.scatter(X, Y, s=10)
#         plt.xlim(0, 1)
#         plt.ylim(0, 1)
#         plt.show()
# =============================================================================
###############################################################################