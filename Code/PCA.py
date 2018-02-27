'''
Created on Sep 25, 2017

@author: opensam - Shubham Sharma
'''

import numpy as np
from numpy import linalg as LA
import pandas as pd
from ggplot import *
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

input_files = ['pca_a.txt','pca_b.txt','pca_c.txt']

''' Credit for the plot : https://stackoverflow.com/questions/21654635/scatter-plots-in-pandas-pyplot-how-to-plot-by-category '''    

def plot_graphs(l1, l2, y, name, algorithm):
	df = pd.DataFrame(dict(x=l1, y=l2, label=y))
	g = ggplot(aes(x='x', y='y', color='label'), data=df) + geom_point(size=50) + theme_bw()
	g.save(name+"_"+algorithm+".jpg")


for input_file in input_files:      
	_df = pd.read_table(input_file, delimiter='\t', header = None)
	X = _df.ix[:,0:3].values
	y = _df.ix[:,4].values
	X_normalized = StandardScaler().fit_transform(X)

	#self implemented pca
	cov_mat = np.cov(X_normalized.T) #np.corrcoef(X_normalized.T)
	val, vec = np.linalg.eig(cov_mat)
	eig = []
	for i in range(len(val)):
		eig.append((np.abs(val[i]), vec[:,i]))
	eig.sort(key=lambda x: x[0], reverse=True)
	d1 = eig[0][1].reshape(4,1)
	d2 = eig[1][1].reshape(4,1)
	y_d = np.hstack((d1,d2))
	Y = X_normalized.dot(y_d)
	plot_graphs(Y[:,0],Y[:,1], y, input_file[:-4],"pca")
	input_tsne = TSNE(n_components=2).fit_transform(X)
	#print(input_tsne.shape)
	plot_graphs(input_tsne[:,0],input_tsne[:,1], y, input_file[:-4],"tsne")

	U, s, V = LA.svd(X, full_matrices=True)
	S = np.zeros((U.shape[0], V.shape[0]), dtype=complex)
	S[:V.shape[0], :V.shape[0]] = np.diag(s)
	#S[:2, :2] = np.diag(s)
	np.allclose(X, np.dot(U, np.dot(S, V)))
	input_svd = np.dot(U, np.dot(S, V))
	plot_graphs(input_svd[:,0],input_svd[:,1], y, input_file[:-4],"svd")
