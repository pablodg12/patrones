##Tarea 1 Patrones Parte 2

import numpy as np
import copy
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import sklearn as sc


def generate_data(dataname,n,noise):
    np.random.seed(1153)
    if dataname == "swissRoll":
        X = np.zeros((n,3))
        t = (3*np.pi/2)*(1 + 2*np.random.uniform(0,1,n))
        height = 21*np.random.uniform(0,1,n)
        X[:,0] = t*np.cos(t) + noise*np.random.normal(0,1,n)
        X[:,1] = height + noise*np.random.normal(0,1,n)
        X[:,2] = t*np.sin(t) + noise*np.random.normal(0,1,n)
        labels = t.astype('uint8') 
        return(X,labels)

    if dataname == "brokenswiss":
        X = np.zeros((n,3))
        a = (3*np.pi/2)*((1 + 2*np.random.uniform(0,1,np.ceil(n/2))*0.4))
        b = (3*np.pi/2)*(1 + 2*(0.4*np.random.uniform(0,1,np.floor(n/2))+ 0.6))
        T = np.concatenate([a,b],axis=0)
        height = 21*np.random.uniform(0,1,n)
        X[:,0] = T*np.cos(T) + noise*np.random.normal(0,1,n)
        X[:,1] = height + noise*np.random.normal(0,1,n)
        X[:,2] = T*np.sin(T) + noise*np.random.normal(0,1,n)
        labels = T.astype('uint8') 
        return(X,labels)
        
    if dataname == "helix":
        X = np.zeros((n,3))
        t = (np.arange(1,n+1)*2*np.pi/n)
        X[:,0] = (2+np.cos(8+t))*np.cos(t)+noise*np.random.normal(0,1,n)
        X[:,1] = (2+np.cos(8*t))*np.sin(t)+noise*np.random.normal(0,1,n)
        X[:,2] = np.sin(8*t)+noise*np.random.normal(0,1,n)
        labels = t.astype('uint8')
        return(X,labels)
        
    if dataname == "intersect":
        X = np.zeros((n,3))
        t = (np.arange(1,n+1)/n*2*np.pi)
        x = np.cos(t)
        y = np.sin(t)
        height = np.random.uniform(0,1,len(x)) * 5
        X[:,0] = x + noise*np.random.normal(0,1,n)
        X[:,1] =x*y + noise*np.random.normal(0,1,n)
        X[:,2] =height + noise*np.random.normal(0,1,n)   
        labels = t.astype('uint8')
        return(X,labels)
    else:
        print("Argumento invalido")
        return 0


def missing_value(porcentaje,data):
    contador = 0
    r = copy.copy(data)
    M = np.zeros((porcentaje*3000,2))
    while(contador != porcentaje*3000):
        i = int(np.round(np.random.uniform(0,2999,1)))
        k = int(np.round(np.random.uniform(0,2,1)))
        if(r[i,k] != np.nan):
            r[i,k] = np.nan
            M[contador,0] = i
            M[contador,1] = k
            contador = contador + 1
    return(r,M)
        
def zero_imputation(missing_matrix,data):
    j = copy.copy(data)
    for k in missing_matrix:
        j[k[0],k[1]] = 0
    return(j)

def mean_imputation(missing_matrix,data):
    h = copy.copy(data)
    mean = np.nanmean(data,axis=0)
    for t in missing_matrix:
        h[t[0],t[1]] = mean[t[1]]
    return(h)


def svd_imputation_label(missing_matrix,data,tol,labels,joined):
    input_matrix = copy.copy(data)
    mean_matrix = mean_label_imputation(joined_dataset,input_matrix,missing_matrix,labels)
    estimated_data = svd_imputation_processation(missing_matrix,mean_matrix)
    while(np.linalg.norm(mean_matrix - estimated_data)/np.linalg.norm(mean_matrix)> tol):
        mean_matrix = copy.copy(estimated_data)
        estimated_data = svd_imputation_processation(missing_matrix,estimated_data)
    return(estimated_data)
    
def svd_imputation(missing_matrix,data,tol):
    input_matrix = copy.copy(data)
    mean_matrix = mean_imputation(missing_matrix,input_matrix)
    estimated_data = svd_imputation_processation(missing_matrix,mean_matrix)
    while(np.linalg.norm(mean_matrix - estimated_data)/np.linalg.norm(mean_matrix)> tol):
        mean_matrix = copy.copy(estimated_data)
        estimated_data = svd_imputation_processation(missing_matrix,estimated_data)
    return(estimated_data)
    
def svd_imputation_processation(missing_matrix,data):
    z = copy.copy(data)
    U_3, d_3, V_3 = np.linalg.svd(z, full_matrices = True)
    esp = np.zeros((abs(data.shape[0]-data.shape[1]),min(z.shape[0],z.shape[1]))) 
    d_3 = np.concatenate((np.diag(d_3), esp))
    V_2 = copy.copy(V_3)
    V_2[2,] = 0
    V_1 = copy.copy(V_2)
    V_1[1,] = 0
    norm_one = np.linalg.norm(z - np.dot(U_3,np.dot(d_3,V_2)))/np.linalg.norm(z)
    norm_two = np.linalg.norm(z - np.dot(U_3,np.dot(d_3,V_1)))/np.linalg.norm(z)
    if(norm_one < norm_two):
        for t in missing_matrix:
            z[t[0],t[1]] = np.dot(U_3,np.dot(d_3,V_2))[t[0],t[1]]
        return(z)
    else:
        for t in missing_matrix:
            z[t[0],t[1]] = np.dot(U_3,np.dot(d_3,V_1))[t[0],t[1]]
        return(z)

def knn_imputation(dataset,missing,k):
    a = copy.copy(dataset)
    b = copy.copy(missing)
    imputed_mean_data = mean_imputation(b,a)
    missing_data = []
    for r in b:
        missing_data.append(imputed_mean_data[r[0],])
    missing_data = np.array(missing_data)
    nbrs = NearestNeighbors(k, algorithm = "ball_tree").fit(imputed_mean_data)
    distances, indices = nbrs.kneighbors(missing_data)
    for t in range(0,indices.shape[0]):
        y = b[t,]
        r = np.mean(imputed_mean_data[indices[t,1:indices.shape[1]],y[1]])
        a[y[0],y[1]] = r
    return(a)
    
def mean_label_imputation(dataset_label,dataset,missing,labels):
    a = copy.copy(dataset_label)
    b = copy.copy(dataset)
    acumulada = 0
    lista = []
    mean = []
    for k in range(0,256):
        lista.append(sum(a[:,3] == k))
    for z in lista:
        if z != 0:
            mean.append(np.nanmean(a[acumulada:acumulada+z-1,0:3],axis=0))
            acumulada = z + acumulada
        if z == 0:
            mean.append(0)
    for h in missing:
        b[h[0],h[1]] = mean[int(labels[h[0]])][int(h[1])]
    return(b)
    

dataset,labels1 = generate_data("swissRoll",3000,0.05)
dataset1, missing = missing_value(0.1,dataset)
joined_dataset = np.c_[dataset1,labels1]
joined_dataset = joined_dataset[np.argsort(joined_dataset[:,3])]
zero_data = zero_imputation(missing,dataset)
mean_data = mean_imputation(missing,dataset)
label_mean_data=mean_label_imputation(joined_dataset,dataset1,missing,labels1)
svd_data = svd_imputation(missing,dataset1,0.02)
svd_data_label = svd_imputation_label(missing,dataset1,0.02,labels1,joined_dataset)
knn_data = knn_imputation(dataset1,missing,25)

print(np.linalg.norm(dataset))
print(np.linalg.norm(svd_data))
print(np.linalg.norm(knn_data))
print(np.linalg.norm(mean_data))
print(np.linalg.norm(zero_data))
print(np.linalg.norm(label_mean_data))
print(np.linalg.norm(svd_data_label))




