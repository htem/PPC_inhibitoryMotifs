from numpy import *
from numpy.linalg import norm, inv
import scipy.io
from matplotlib.pyplot import *
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
from scipy.ndimage import gaussian_filter
from numpy.linalg import eigvalsh, qr
from rc_parameters import *
from scipy.special import expit
import pandas as pd
import pickle
from sklearn.svm import SVC
from scipy.stats import multivariate_normal
from sklearn.svm import LinearSVC
from scipy.io import loadmat
from scipy.stats import ttest_ind, mannwhitneyu
# from scipy.stats import permutation_test
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import cross_val_score
from scipy.io import savemat, loadmat