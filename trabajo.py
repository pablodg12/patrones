#Tarea 1 Reconocimiento de Patrones Parte 1 
import numpy as np
import sklearn as sc
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.feature_selection import VarianceThreshold
from sklearn import manifold
from sklearn import decomposition 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

def graficar2d(data,labels,inf,sup,dim):
    if dim == 1:
        y = np.zeros(len(data))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(inf, sup)
        ax.set_ylim(inf, sup)
        ax.scatter(data, y, c= labels, s= 20)
        plt.show()
    elif dim == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(inf, sup)
        ax.set_ylim(inf, sup)
        ax.scatter(data[:,0], data[:,1], c= labels, s= 20)
        plt.show()
    else:
        print("Invalid Dimension")

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

        

#graficar3d(data1,labels1,-15,15,data2,labels2,-12,12,data3,labels3,-2,2,data4,labels4,-2,2)  


#Figuras 3d, funcion para graficar incialemnte la data creada

def graficar_3d():
    
    data1,labels1 = generate_data("swissRoll",3000,0.05)
    data2,labels2 = generate_data("brokenswiss",3000,0.05)
    data3,labels3 = generate_data("helix",3000,0.05)
    data4,labels4 = generate_data("intersect",3000,0.05)    
    
    fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_zlim(-15, 15)
    ax.scatter(data1[:,0], data1[:,1], data1[:,2], c=labels1, s= 20)
    
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.set_xlim(-15, 15)
    ax2.set_ylim(-15, 15)
    ax2.set_zlim(-15, 15)
    ax2.scatter(data2[:,0], data2[:,1], data2[:,2], c=labels2, s= 20)
    
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-2, 2)
    ax3.set_zlim(-2, 2)
    ax3.scatter(data3[:,0], data3[:,1], data3[:,2], c=labels3, s= 20)
    
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.set_xlim(-2, 2)
    ax4.set_ylim(-2, 2)
    ax4.set_zlim(-2, 2)
    ax4.scatter(data4[:,1], data4[:,2], data4[:,0], c=labels4, s= 20)
    
    plt.show()
 
 
def Gauss_Kernel(x1,x2,gamma):
    return(np.exp(-gamma*np.linalg.norm(np.subtract(x1,x2))**2))

def gauss_Kernel_matrix(data,gamma):
    matrix = np.zeros((data.shape[0],data.shape[0]))
    data = data 
    for i in range(0,data.shape[0]):
        for t in range(0,data.shape[0]):
            matrix[t,i] = Gauss_Kernel(data[i,:],data[t,:],gamma)
    return(matrix)

                    #Plotear usando kernel PCA para 2 y 1 dimension#   
#Figura puede tomar valores como "swissRoll", "brokenswiss","helix","intersect"
#kernel "lineal" para usar PCA
#kernel "rbf" para gaussiana(utiliza parametro gamma), "poly" para polinomial (utiliza parametros gamma y libertad)
    
def plot2d_1d_KPCA(figure,dim,kernel_function,gammavalue,libertad):
    
    if figure == "swissRoll":     
        data1,labels1 = generate_data("swissRoll",3000,0.05)
        kpca = sc.decomposition.KernelPCA(dim ,kernel=kernel_function, gamma = gammavalue, degree = libertad)
        X_fit_transformed = kpca.fit_transform(data1)
        graficar2d(X_fit_transformed,labels1,-10000,10000,dim)
    if figure == "brokenswiss":
        data1,labels1 = generate_data("brokenswiss",3000,0.05)
        kpca = sc.decomposition.KernelPCA(dim ,kernel=kernel_function, gamma = gammavalue, degree = libertad)
        X_fit_transformed = kpca.fit_transform(data1)
        graficar2d(X_fit_transformed,labels1,-1,1,dim) 
    if figure == "helix":
        data1,labels1 = generate_data("helix",3000,0.05)
        kpca = sc.decomposition.KernelPCA(dim ,kernel=kernel_function, gamma = gammavalue, degree = libertad)
        X_fit_transformed = kpca.fit_transform(data1)
        graficar2d(X_fit_transformed,labels1,-1,1,dim)
    if figure == "intersect":
        data1,labels1 = generate_data("intersect",3000,0.05)
        kpca = sc.decomposition.KernelPCA(dim ,kernel=kernel_function, gamma = gammavalue, degree= libertad)
        X_fit_transformed = kpca.fit_transform(data1)
        graficar2d(X_fit_transformed,labels1,-1,1,dim)
        
        
        
        #Dimensionalidad intrisica PCA    
data1,labels1 = generate_data("swissRoll",3000,0.05)
data2 = (data1 - data1.mean(0))/data1.std(0)
#p = np.cov(data1,bias=True)
#U_r, d_r, V_r = np.linalg.svd(p)


    #Dimensionalidad intrisinca KERNEL PCA
#test = gauss_Kernel_matrix(data1,0.05)
#k = np.dot(test,np.transpose(test))/3000
#r = np.cov(test,bias=True)
#U_r1, d_r1, V_r1 = np.linalg.svd(r)

#Graficar lo de arriba

#
#tot = sum(d_r1)
#var_exp = [(i / tot)*100 for i in sorted(d_r1, reverse=True)]
#cum_var_exp = np.cumsum(var_exp)

#with plt.style.context('seaborn-whitegrid'):
#    plt.figure(figsize=(6, 4))

#    plt.bar(range(5), var_exp[0:5], alpha=0.5, align='center',
#            label='individual explained variance')
#    plt.step(range(5), cum_var_exp[0:5], where='mid',
#             label='cumulative explained variance')
#    plt.ylabel('Explained variance ratio')
#    plt.xlabel('Principal components')
#    plt.legend(loc='best')
#    plt.tight_layout()



isomap = sc.manifold.Isomap(100,2)
#pca = sc.decomposition.PCA(2)
#x_pca = pca.fit_transform(data1)
x_isomap = isomap.fit_transform(data1)



#x_mds = sc.manifold.MDS(2)
#x_mds_data = x_mds.fit_transform(data1)


