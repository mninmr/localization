import re
import numpy as np
from scipy.stats import norm as gaussian
import matplotlib.pyplot as plt
import math

def gauss_prob(observed, mean, variance):
    """
    :param observed: an observed value
    :param mean: the mean of the distribution
    :param variance: the variance of the distribution
    :return: the probability of seeing the observation given the mean and variance of gaussian distribution
    """
    return gaussian(mean, math.sqrt(variance)).pdf(observed)

def gauss_sample(mean, variance):
    """
    :param mean: the mean of the distribution
    :param variance: the variance of the distribution
    :return: a random sample from a gaussian with the specified parameters
    """
    return np.random.normal(mean, math.sqrt(variance))

def init_vis(gmap, states, true_state):
    """
    initializes visual display with a set of initial points
    :param gmap: a grayscale map
    :param states: an array x,y,phi arrays
    :param true_state: the current true state as given by the ground.csv file
    """
    plt.ion()
    update_vis(gmap, states, true_state)


def update_vis(gmap, states, true_state):
    """
    updates the already initialized visual display with new points
    :param gmap: a grayscale map
    :param states: an array x,y,phi arrays
    :param true_state: the current true state as given by the ground.csv file
    """
    plt.clf()
    h,w = gmap.shape
    plt.imshow(gmap, cmap='Greys')
    plt.scatter(states[:,0], states[:,1], c='cyan')
    plt.scatter(true_state[0], true_state[1], c='red')
    plt.draw()

def load_map(fname):
    """
    :param fname: - the name of the raw pgm file to load
    :return: a matrix form of the map, 0 for unoccupied, 1 for occupied
    """
    image = read_pgm(fname, byteorder='<')
    uni = np.unique(image)
    if(len(uni) == 3): #the strange format of lidar data
        #image = np.logical_and(image != np.min(uni), image != np.max(uni)) #min is wall, max is unknown
        image = image != np.max(uni)
        image = np.array(image, dtype=int, ndmin=2)
    elif(not (len(uni) == 2 and 0 in uni and 1 in uni)): #if it is not already a 0,1 array
        image = image > 150
        image = np.array(image, dtype=int, ndmin=2)
    image = fulltrim(image)
    return image

def load_csv(fname):
    """
    :param fname: a csv filename to load into a numpy array
    :return: the csv file loaded into a NUMLINES x NUMELEM array
    This is for loading starting position and control csv files
    """
    return np.loadtxt(fname, delimiter=",", ndmin=2)

def load_measurements(fname):
    """
    :param fname: a file name for a measurements csv file to parse
    :return: a TxMx2 numpy array.
    This is for loading a measurements file.
    Each line contains comma separated pairs of degree and distance, which are separated by a colon
    """
    with open(fname) as f:
        content = f.readlines()
        return np.array(map(lambda(line): map(lambda(pair): map(lambda(elem): float(elem),pair.split(":")), line.split(",")), content))
"""
The following are helper functions and not necessary to understand the project
"""

def read_pgm(filename, byteorder='>'):
    """
    Return image data from a raw PGM file as numpy array.
    Format specification: http://netpbm.sourceforge.net/doc/pgm.html
    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


def trim(figure):
    ret = figure
    for i in range(len(figure)):
        if 0 in figure[i]:
            ret = figure[i:]
            break
    l = len(ret) - 1
    for i in range(l + 1):
        if 0 in ret[(l - i)]:
            ret = ret[:(l-i)]
            break
    return ret


def fulltrim(map):
    """
    trims all but one layer of empty space off the given map
    """
    map = trim(trim(map.T).T)
    h, w = map.shape
    map2 = np.concatenate([
        np.ones((1, w)),
        map,
        np.ones((1, w))], axis=0)

    map3 = np.concatenate([
        np.ones((h+2, 1)),
        map2,
        np.ones((h+2, 1))], axis=1)

    return map3
