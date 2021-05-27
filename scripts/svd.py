import pandas as pd
from glob import glob
import numpy as np
import os, sys

from numpy.matlib import repmat
from sklearn import preprocessing

def run_svd(all_data, cut):
    U, s, V = np.linalg.svd(np.transpose(all_data), full_matrices=True)
    # Valores singulares normalizados
    y = s * (1/np.sum(s))
    # Calcula o "joelho" da curva
    if cut > 0:
        rank = cut
    else:
        rank = get_knee(y)

    # Computa matrix aproximada de posto reduzido
    new_data = low_rank_approximation(s, V , rank)
    # Normalizando os dados
    data_scaled = preprocessing.scale(new_data)
    # Feature Selection
    #data_transformed = VarianceThreshold().fit_transform(data_scaled)
    # separa train_set do test_set

    return data_scaled

'''
  Funções auxiliares para o SVD
'''
# In: singular values
def get_knee(sgl_values):
    values = list(sgl_values)

    #get coordinates of all the points
    nPoints = len(values)
    allCoord = np.vstack((range(nPoints), values)).T

    # get the first point
    firstPoint = allCoord[0]

    # get vector between first and last point - this is the line
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    # find the distance from each point to the line:
    # vector between all points and first point
    vecFromFirst = allCoord - firstPoint
    ''' To calculate the distance to the line, we split vecFromFirst into two
    components, one that is parallel to the line and one that is perpendicular
    Then, we take the norm of the part that is perpendicular to the line and
    get the distance.

    We find the vector parallel to the line by projecting vecFromFirst onto
    the line. The perpendicular vector is vecFromFirst - vecFromFirstParallel
    We project vecFromFirst by taking the scalar product of the vector with
    the unit vector that points in the direction of the line (this gives us
    the length of the projection of vecFromFirst onto the line). If we
    multiply the scalar product by the unit vector, we have vecFromFirstParallel
    '''
    scalarProduct = np.sum(vecFromFirst * repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel

    # distance to line is the norm of vecToLine
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))

    # knee/elbow is the point with max distance value
    return np.argmax(distToLine) + 1

# In: rank, sigle values and rows
def low_rank_approximation(s, V, rank):
    return np.transpose(np.dot(np.diag(s[:rank]), V[:][:rank]))

if __name__ == '__main__':
    data_file = sys.argv[1]
    data = pd.read_csv(data_file, index_col='name')
    new_data = run_svd(data.iloc[:,:-1], -1)
    df = pd.DataFrame(new_data, index=data.index)
    df['class'] = data['class']
    df.to_csv(os.path.dirname(data_file) + os.sep + 'data_svd.csv')
