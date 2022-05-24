# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 19:50:21 2020

@author: Cristian Gonzalez
"""
# ruta: ubicacion de donde se encutran las imagenes
# Numero: devuelve la cantidad de imagnes que hay en la ruta
def read(ruta):
    import glob
    Numero=len(glob.glob(ruta))
    print(str(Numero))
    return Numero

# bh: cantidad de datos en filas
# bv: cantidad de datos en columnas
# can: cantidad de imagenes que se van analizar
# ir: ruta de donde se encutran las imagenes
# c: devuelve las imagenes remuestradas y apiladas
def matriz(bh,bv,can,ir):

    from skimage import io
    import math
    import numpy as np
    
    c=np.zeros((bh-1,bv-1,3))
    c2=np.zeros((bh-1,bv-1,3))
    
    #print(ir+str(1)+".tif")
    image=io.imread(ir+"1.tif")/255.0
    alto,ancho,dimension=image.shape
    sv=ancho/bv
    sh=alto/bh
    for D in range(0,dimension):
        for X in range(1,bh):
            for Y in range(1,bv):
                c[X-1,Y-1,D]=image[math.floor(sh*X),math.floor(sv*Y),D]
                
    for M in range(1,can):
        
        #print(ir+str(M+1)+".tif")
        image=io.imread(ir+str(M+1)+".tif")/255.0
        alto,ancho,dimension=image.shape
        sv=ancho/bv
        sh=alto/bh    
        
        for D in range(0,dimension):
            for X in range(1,bh):
                for Y in range(1,bv):
                    c2[X-1,Y-1,D]=image[math.floor(sh*X),math.floor(sv*Y),D]
                    
        c=np.concatenate((c,c2))

    return c

# bh: cantidad de datos en filas
# bv: cantidad de datos en columnas
# N: cantidad de imagenes que se van analizar
# ir: ruta de donde se encutran las imagenes    
# c: devuelve las imagenes remuestradas y apiladas
def matriz_test(bh,bv,N,ir):

    from skimage import io
    import math
    import numpy as np
    
    c=np.zeros((bh-1,bv-1,3))
    
    #print(ir+str(N+1)+".tif")
    image=io.imread(ir+str(N+1)+".tif")/255.0
    alto,ancho,dimension=image.shape
    sv=ancho/bv
    sh=alto/bh
    for D in range(0,dimension):
        for X in range(1,bh):
            for Y in range(1,bv):
                c[X-1,Y-1,D]=image[math.floor(sh*X),math.floor(sv*Y),D]

    return c

# N: la cantidad de etiquetas que es igual al numero de cluster
# etiqueta: los datos obtenidos de la prediccion del km
# datos: las imagenes en rgb
# conteo: devuelve la cuenta de la insidencia de las etiquetas para formar el histograma
# cluster: devuelve la cantidad de clusters
def contar(N,etiqueta,datos):
    
    import numpy as np
    
    cluster=np.arange(N)
    conteo=np.zeros((len(cluster)))
    v=0
    for Y in range(0,len(cluster)):
        for X in range(0,len(etiqueta)):
            if etiqueta[X]==cluster[Y]:
                conteo[Y]=conteo[Y]+1
                v=v+1
                
    return conteo,cluster

# datos: matriz rgb 
# X: devuelve una matricNx3 rgb
def acomodar(datos):
    
    import numpy as np
    
    X1,Y1,Z1=datos.shape
    rojo=np.zeros((X1*Y1))
    verde=np.zeros((X1*Y1))
    azul=np.zeros((X1*Y1))
    
    for col in range(0,Y1):
        for fil in range(0,X1):
            rojo[(col*X1)+fil]=datos[fil,col,0]
            #print(str((col*X1)+fil))
            verde[(col*X1)+fil]=datos[fil,col,1]
            azul[(col*X1)+fil]=datos[fil,col,2]
    
    X=np.c_[rojo,verde,azul]
    
    return X

# datos: tres columnas rojo verde y azul
# N: la cantidad de cluster
def km(datos,N):
    
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from mpl_toolkits.mplot3d import Axes3D
    
    kmeans = KMeans(n_clusters=N, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(datos)
    centroids = kmeans.cluster_centers_
    #print(centroids)
    
    
    # Predicting the clusters
    labels = kmeans.predict(datos)
    # Getting the cluster centers
    #C = kmeans.cluster_centers_
    #colores=['red','green','blue']
    #asignar=[]
    #for row in labels:
    #    asignar.append(colores[row])
    
    #fig = plt.figure()
    #ax = Axes3D(fig)
    #ax.scatter(datos[:, 0], datos[:, 1], datos[:, 2], c=asignar,s=60)
    #ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000)

    return labels, centroids

# datos rgb apilados
# dnn: devuelve la matriz rgb sin fondo
def quitar_negros(datos):
    
    import numpy as np
    
    v=0
    dn=np.zeros((len(datos), 3))
    for X in range(0,len(datos)):
        if datos[X,0]!=0 and datos[X,1]!=0 and datos[X,2]!=0:            
            for Y in range(0,3):            
                dn[v,Y]=datos[X,Y]
            v=v+1
            
    dnn=dn[range(0,v),:]
    
    return dnn

# datos: conteo de los datos rgb
def graficar(datos):
    
    import matplotlib.pyplot as plt
    
    fig = plt.figure(u'Gráfica de barras') # Figure
    ax = fig.add_subplot(111) # Axes
    
    xx = range(len(datos))
    
    ax.bar(xx, datos, width=0.8, align='center')
    ax.set_xticks(xx)
    ax.set_xticklabels(xx)
    
    plt.show()
    
def codigo(datosN,cluster):
    
    from time import time,localtime
    
    actual=localtime()
    print("            codigo "+str(actual[3])+":"+str(actual[4])) 
    ti = time()
    print("cluster "+str(cluster))
    labels,centroide=km(datosN,cluster)
    #print("Conteo de cada uno de los grupos")
    conteo,nombres=contar(cluster,labels,datosN)
    tiempo=(time()-ti)/60
    print("Tiempo de bag of word cluster "+str(cluster)+" \n "+str(tiempo)+" minutos")
    
    return conteo,centroide,labels,tiempo

# X1_train: magnitudes de cada una de las flores
# y1_train: etiquetas de las magnitudes
# X1_test: datos de validacion
# y1_test: etiqueta de cada una de las flores de validacion
# scores: exactud entre los datos 
def KNN2(X1_train,y1_train,X1_test,y1_test):
    
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier
    from time import time,localtime    
    import numpy as np    
    from sklearn.metrics import confusion_matrix
    
    actual=localtime()
    print("            codigo "+str(actual[3])+":"+str(actual[4])) 
    ti = time()
    
    
    confusion=np.zeros((4,4,30))
    scores = []#promedio de la exactutud
    #ajuste
    max_k=30
    for k in range(1, max_k):
        # create knn classifier with k = k
        knn = KNeighborsClassifier(n_neighbors=k)
        # train the model
        knn.fit(X1_train, y1_train)
        # predict labels for test samples
        y1_pred = knn.predict(X1_test)
          
        # add accuracy to score table
        scores.append(accuracy_score(y1_test, y1_pred))
        #print('vecino: '+str(k)+' '+str(y1_pred))
        confusion[:,:,k]=confusion_matrix(y1_test, y1_pred)
    
    tiempo=(time()-ti)*1000
    print("Tiempo knn    "+str(tiempo)+" mili-segundos")      
    return scores, tiempo,confusion

#def svm():
    

# X1_train: centroides
# y1_train: etiquetas de los centroides
# X1_test: datos rgb
# y1_pred: vecino mas cercano a los centroides
def KNN(X1_train,y1_train,X1_test):
    
    from sklearn.neighbors import KNeighborsClassifier
    
    from sklearn.metrics import confusion_matrix
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X1_train, y1_train)
    y1_pred=knn.predict(X1_test)
    
    #confusion=confusion_matrix(y1_test, et_v)
    
    return y1_pred

def svm1(X_train,Y_train,X_prueba,Y_prueba):
    
    from sklearn import svm
    from sklearn.metrics import accuracy_score
    from time import time,localtime
    
    actual=localtime()
    print("            codigo "+str(actual[3])+":"+str(actual[4])) 
    ti = time()
    
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, Y_train)
    y_pred=clf.predict(X_prueba)
    sal= (accuracy_score(Y_prueba, y_pred))
    
    tiempo=(time()-ti)*1000
    print("Tiempo SVM_1    "+str(tiempo)+" mili-segundos") 
    return sal,tiempo

def svm2(X_train,Y_train,X_prueba,Y_prueba):
    
    from sklearn import svm
    from sklearn.metrics import accuracy_score
    from time import time,localtime
    
    actual=localtime()
    print("            codigo "+str(actual[3])+":"+str(actual[4])) 
    ti = time()
    
    clf = svm.LinearSVC(max_iter=10000)
    clf.fit(X_train, Y_train)
    y_pred=clf.predict(X_prueba)
    sal= (accuracy_score(Y_prueba, y_pred))
    
    tiempo=(time()-ti)*1000
    print("Tiempo SVM_2    "+str(tiempo)+" mili-segundos") 
    return sal,tiempo

def svm3(X_train,Y_train,X_prueba,Y_prueba):
    
    from sklearn import svm
    from sklearn.metrics import accuracy_score
    from time import time,localtime
    
    actual=localtime()
    print("            codigo "+str(actual[3])+":"+str(actual[4])) 
    ti = time()
    
    clf = svm.SVC(kernel='rbf', gamma=0.7)
    clf.fit(X_train, Y_train)
    y_pred=clf.predict(X_prueba)
    sal= (accuracy_score(Y_prueba, y_pred))
    
    tiempo=(time()-ti)*1000
    print("Tiempo SVM_3    "+str(tiempo)+" mili-segundos") 
    return sal,tiempo

def svm4(X_train,Y_train,X_prueba,Y_prueba,grado):
    
    from sklearn import svm
    from sklearn.metrics import accuracy_score
    from time import time,localtime
    
    actual=localtime()
    print("            codigo "+str(actual[3])+":"+str(actual[4])) 
    ti = time()
    
    sal = []
    for k in range(0, grado):
        clf = svm.SVC(kernel='poly', degree=k, gamma='auto')#variar degree 4
        clf.fit(X_train, Y_train)
        y_pred=clf.predict(X_prueba)
        sal.append(accuracy_score(Y_prueba, y_pred))
    
    tiempo=(time()-ti)*1000
    print("Tiempo SVM_4    "+str(tiempo)+" mili-segundos") 
    return sal,tiempo

def ANN(X_train,Y_train,X_prueba,Y_prueba,neurona):
    
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    import numpy as np
    from time import time,localtime
    
    actual=localtime()
    print("            codigo "+str(actual[3])+":"+str(actual[4])) 
    ti = time()
    
    N=0
    a=np.zeros(5)
    sal1=np.zeros((int(neurona/10),5))
    sal2=np.zeros((int(neurona/10),5))
    sal3=np.zeros((int(neurona/10),5))
    sal4=np.zeros((int(neurona/10),5))
    sal5=np.zeros((int(neurona/10),5))
    
    for x in range(0,int(neurona/10)):
        N=N+10
        alfa_min=1e-5
        for y in range(0,5):
            clf = MLPClassifier(solver='lbfgs', alpha=alfa_min, hidden_layer_sizes=(N,),max_iter=5000)# ,max_iter=5000,random_state=1
            clf.fit(X_train, Y_train)
            y_pred=clf.predict(X_prueba)
            a[y]=(accuracy_score(Y_prueba, y_pred))
            alfa_min=alfa_min*100
            
        sal1[x,:]=a[:]    
    print('paso 1')
    
    N=0
    a=np.zeros(5)
    for x in range(0,int(neurona/10)):
        N=N+10
        alfa_min=1e-5
        for y in range(0,5):
            clf = MLPClassifier(solver='lbfgs', alpha=alfa_min, hidden_layer_sizes=(N,N),max_iter=5000)
            clf.fit(X_train, Y_train)
            y_pred=clf.predict(X_prueba)
            a[y]=(accuracy_score(Y_prueba, y_pred))
            alfa_min=alfa_min*100
            
        sal2[x,:]=a[:]        
    print('paso 2')
    
    N=0
    a=np.zeros(5)
    for x in range(0,int(neurona/10)):
        N=N+10
        alfa_min=1e-5
        for y in range(0,5):
            clf = MLPClassifier(solver='lbfgs', alpha=alfa_min, hidden_layer_sizes=(N,N,N),max_iter=5000)
            clf.fit(X_train, Y_train)
            y_pred=clf.predict(X_prueba)
            a[y]=(accuracy_score(Y_prueba, y_pred))
            alfa_min=alfa_min*100
            
        sal3[x,:]=a[:]    
    print('paso 3')
    
    N=0
    a=np.zeros(5)
    for x in range(0,int(neurona/10)):
        N=N+10
        alfa_min=1e-5
        for y in range(0,5):
            clf = MLPClassifier(solver='lbfgs', alpha=alfa_min, hidden_layer_sizes=(N,N,N,N),max_iter=5000)
            clf.fit(X_train, Y_train)
            y_pred=clf.predict(X_prueba)
            a[y]=(accuracy_score(Y_prueba, y_pred))
            alfa_min=alfa_min*100
            
        sal4[x,:]=a[:]    
    print('paso 4')
    
    N=0
    a=np.zeros(5)
    for x in range(0,int(neurona/10)):
        N=N+10
        alfa_min=1e-5
        for y in range(0,5):
            clf = MLPClassifier(solver='lbfgs', alpha=alfa_min, hidden_layer_sizes=(N,N,N,N,N),max_iter=5000)
            clf.fit(X_train, Y_train)
            y_pred=clf.predict(X_prueba)
            a[y]=(accuracy_score(Y_prueba, y_pred))
            alfa_min=alfa_min*100
            
        sal5[x,:]=a[:]
    print('paso 5')
    
    tiempo=(time()-ti)/60
    print("Tiempo ANN    "+str(tiempo)+" minutos") 
    return sal1,sal2,sal3,sal4,sal5,tiempo

def entrenamiento(f1,f2,f3,f4,labelsN,clusterN):

    import math
    import numpy as np
    from time import time,localtime
    ruta='D:/Maestria/Tesis 3/'
    
    actual=localtime()
    print("            entrenamiento "+str(actual[3])+":"+str(actual[4])) 
    ti = time()
    
    aux=0
    conteo=np.zeros((math.floor(f1*0.6)+math.floor(f2*0.6)+math.floor(f3*0.6)+math.floor(f4*0.6),clusterN))
    et=np.zeros((math.floor(f1*0.6)+math.floor(f2*0.6)+math.floor(f3*0.6)+math.floor(f4*0.6)))
    
    for X in range(0,len(conteo)):
        
        if X<math.floor(f1*0.6):
            h_train=matriz_test(320,120,X,str(ruta)+'/imagenes/muestras python/f1/')
            et[X]=0
        
        if X>=math.floor(f1*0.6) and X<math.floor(f1*0.6)+math.floor(f2*0.6):
            h_train=matriz_test(320,120,X-math.floor(f1*0.6),str(ruta)+'/imagenes/muestras python/f2/')
            et[X]=1
            
        if X>=math.floor(f1*0.6)+math.floor(f2*0.6) and X<math.floor(f1*0.6)+math.floor(f2*0.6)+math.floor(f3*0.6):
            h_train=matriz_test(320,120,X-math.floor(f1*0.6)-math.floor(f2*0.6),str(ruta)+'/imagenes/muestras python/f3/')
            et[X]=2
            
        if X>=math.floor(f1*0.6)+math.floor(f2*0.6)+math.floor(f3*0.6) and X<math.floor(f1*0.6)+math.floor(f2*0.6)+math.floor(f3*0.6)+math.floor(f4*0.6):
            h_train=matriz_test(320,120,X-math.floor(f1*0.6)-math.floor(f2*0.6)-math.floor(f3*0.6),str(ruta)+'/imagenes/muestras python/f4/')
            et[X]=3
            
        d_train=acomodar(h_train)
    
        datosN_test=quitar_negros(d_train)
        
        labelsN_test=labelsN[range(aux,len(datosN_test)+aux)]
        aux=aux+len(datosN_test)
        conteo[X,:],cluster=contar(clusterN,labelsN_test,datosN_test)
    
    tiempo=(time()-ti)/60
    print("Tiempo de entrenamiento cluster "+str(clusterN)+" \n "+str(tiempo)+" minutos")

    return et, conteo,tiempo

def validacion(f1,f2,f3,f4,C):
    
    import math
    import numpy as np
    from time import time,localtime
    ruta='D:/Maestria/Tesis 3/'
    
    actual=localtime()
    print("            validacion "+str(actual[3])+":"+str(actual[4])) 
    ti = time()
    
    conteop=np.zeros((math.ceil((f1-math.floor(f1*0.6))/2)+math.ceil((f2-math.floor(f2*0.6))/2)+math.ceil((f3-math.floor(f3*0.6))/2)+math.ceil((f4-math.floor(f4*0.6))/2),len(C)))
    etp=np.zeros((math.ceil((f1-math.floor(f1*0.6))/2)+math.ceil((f2-math.floor(f2*0.6))/2)+math.ceil((f3-math.floor(f3*0.6))/2)+math.ceil((f4-math.floor(f4*0.6))/2)))
    
    for X in range(0,len(conteop)):
        
        if X<math.ceil((f1-math.floor(f1*0.6))/2):
            h_train=matriz_test(320,120,X+math.floor(f1*0.6),str(ruta)+'/imagenes/muestras python/f1/')
            etp[X]=0
        
        if X>=math.ceil((f1-math.floor(f1*0.6))/2) and X<math.ceil((f1-math.floor(f1*0.6))/2)+math.ceil((f2-math.floor(f2*0.6))/2):
            h_train=matriz_test(320,120,X-math.ceil((f1-math.floor(f1*0.6))/2)+math.floor(f2*0.6),str(ruta)+'/imagenes/muestras python/f2/')
            etp[X]=1
            
        if X>=math.ceil((f1-math.floor(f1*0.6))/2)+math.ceil((f2-math.floor(f2*0.6))/2) and X<math.ceil((f1-math.floor(f1*0.6))/2)+math.ceil((f2-math.floor(f2*0.6))/2)+math.ceil((f3-math.floor(f3*0.6))/2):
            h_train=matriz_test(320,120,X-math.ceil((f1-math.floor(f1*0.6))/2)-math.ceil((f2-math.floor(f2*0.6))/2)+math.floor(f3*0.6),str(ruta)+'/imagenes/muestras python/f3/')
            etp[X]=2
            
        if X>=math.ceil((f1-math.floor(f1*0.6))/2)+math.ceil((f2-math.floor(f2*0.6))/2)+math.ceil((f3-math.floor(f3*0.6))/2) and X<math.ceil((f1-math.floor(f1*0.6))/2)+math.ceil((f2-math.floor(f2*0.6))/2)+math.ceil((f3-math.floor(f3*0.6))/2)+math.ceil((f4-math.floor(f4*0.6))/2):
            h_train=matriz_test(320,120,X-math.ceil((f1-math.floor(f1*0.6))/2)-math.ceil((f2-math.floor(f2*0.6))/2)-math.ceil((f3-math.floor(f3*0.6))/2)+math.floor(f4*0.6),str(ruta)+'/imagenes/muestras python/f4/')
            etp[X]=3
            
        d_train=acomodar(h_train)
    
        datosN_test=quitar_negros(d_train)
        
        labels_test=KNN(C,np.arange(0,len(C)),datosN_test)
        conteop[X,:],cluster=contar(len(C),labels_test,datosN_test)

    tiempo=(time()-ti)/60
    print("Tiempo de validacion cluster "+str(len(C))+" \n "+str(tiempo)+" minutos")

    return etp, conteop,tiempo

def prueba(f1,f2,f3,f4,C):
    
    import math
    import numpy as np
    from time import time,localtime
    ruta='D:/Maestria/Tesis 3/'
    
    actual=localtime()
    print("            prueba "+str(actual[3])+":"+str(actual[4]))    
    ti = time()
    
    conteov=np.zeros((math.floor((f1-math.floor(f1*0.6))/2)+math.floor((f2-math.floor(f2*0.6))/2)+math.floor((f3-math.floor(f3*0.6))/2)+math.floor((f4-math.floor(f4*0.6))/2),len(C)))
    etv=    np.zeros((math.floor((f1-math.floor(f1*0.6))/2)+math.floor((f2-math.floor(f2*0.6))/2)+math.floor((f3-math.floor(f3*0.6))/2)+math.floor((f4-math.floor(f4*0.6))/2)))
    
    for X in range(0,len(conteov)):
        
        if X<math.floor((f1-math.floor(f1*0.6))/2):
            h_train=matriz_test(320,120,X+math.floor(f1*0.6)+math.ceil((f1-math.floor(f1*0.6))/2),str(ruta)+'/imagenes/muestras python/f1/')
            etv[X]=0
        
        if X>=math.floor((f1-math.floor(f1*0.6))/2) and X<math.floor((f1-math.floor(f1*0.6))/2)+math.floor((f2-math.floor(f2*0.6))/2):
            h_train=matriz_test(320,120,X-math.floor((f1-math.floor(f1*0.6))/2)+math.floor(f2*0.6)+math.ceil((f2-math.floor(f2*0.6))/2),str(ruta)+'/imagenes/muestras python/f2/')
            etv[X]=1
            
        if X>=math.floor((f1-math.floor(f1*0.6))/2)+math.floor((f2-math.floor(f2*0.6))/2) and X<math.floor((f1-math.floor(f1*0.6))/2)+math.floor((f2-math.floor(f2*0.6))/2)+math.floor((f3-math.floor(f3*0.6))/2):
            h_train=matriz_test(320,120,X-math.floor((f1-math.floor(f1*0.6))/2)-math.floor((f2-math.floor(f2*0.6))/2)+math.floor(f3*0.6)+math.ceil((f3-math.floor(f3*0.6))/2),str(ruta)+'/imagenes/muestras python/f3/')
            etv[X]=2
            
        if X>=math.floor((f1-math.floor(f1*0.6))/2)+math.floor((f2-math.floor(f2*0.6))/2)+math.floor((f3-math.floor(f3*0.6))/2) and X<math.floor((f1-math.floor(f1*0.6))/2)+math.floor((f2-math.floor(f2*0.6))/2)+math.floor((f3-math.floor(f3*0.6))/2)+math.ceil((f4-math.floor(f4*0.6))/2):
            h_train=matriz_test(320,120,X-math.floor((f1-math.floor(f1*0.6))/2)-math.floor((f2-math.floor(f2*0.6))/2)-math.floor((f3-math.floor(f3*0.6))/2)+math.floor(f4*0.6)+math.ceil((f4-math.floor(f4*0.6))/2),str(ruta)+'/imagenes/muestras python/f4/')
            etv[X]=3
            
        d_train=acomodar(h_train)
    
        datosN_test=quitar_negros(d_train)
        
        labels10_test=KNN(C,np.arange(0,len(C)),datosN_test)
        conteov[X,:],cluster=contar(len(C),labels10_test,datosN_test)

    tiempo=(time()-ti)/60
    print("Tiempo de prueba cluster "+str(len(C))+" \n "+str(tiempo)+" minutos")

    return etv, conteov,tiempo

# bh: cantidad de datos en filas
# bv: cantidad de datos en columnas
# N: cantidad de imagenes que se van analizar
# ir: ruta de donde se encutran las imagenes    
# c: devuelve las imagenes remuestradas y apiladas
def matriz_interfaz(bh,bv,N,ir):

    from skimage import io
    import math
    import numpy as np
    import cv2 as cv
    
    c=np.zeros((bh-1,bv-1,3))
    
    #print(ir+str(N+1)+".tif")
    #image=io.imread(ir)/255.0
    image=cv.imread(ir)#/255.0
    
    
    alto,ancho,dimension=image.shape
    sv=ancho/bv
    sh=alto/bh
    for D in range(0,3):
        for X in range(1,bh):
            for Y in range(1,bv):
                c[X-1,Y-1,D]=image[math.floor(sh*X),math.floor(sv*Y),D]

    return c

# X1_train: magnitudes de cada una de las flores
# y1_train: etiquetas de las magnitudes
# X1_test: datos de validacion
# y1_test: etiqueta de cada una de las flores de validacion
# scores: exactud entre los datos 
#def KNN_interfaz(X1_train,y1_train,X1_test,y1_test):
def KNN_interfaz(X1_train,y1_train,X1_test,y1_test):
    
    #X1_train=con30_t
    #y1_train=et_t
    #X1_test=conteov
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier
    from time import time,localtime
    
    actual=localtime()
    print("            codigo "+str(actual[3])+":"+str(actual[4])) 
    ti = time()
    
    scores = []#promedio de la exactutud
    #ajuste
    k=7
    
    # create knn classifier with k = k
    knn = KNeighborsClassifier(n_neighbors=k)
    # train the model
    knn.fit(X1_train, y1_train)
    # predict labels for test samples
    y1_pred = knn.predict(X1_test)
          
    # add accuracy to score table
    #scores.append(accuracy_score(y1_test[0], y1_pred[0]))
    
    tiempo=(time()-ti)*1000
    print("Tiempo knn    "+str(tiempo)+" mili-segundos")      
    return y1_pred,tiempo

def prueba_interfaz_KNN(h_train):
    
    import math
    import numpy as np
    from time import time,localtime
    ruta='D:/Maestria/Tesis 3/'
    
    actual=localtime()
    print("            prueba "+str(actual[3])+":"+str(actual[4]))    
    ti = time()
    
    dir1=str(ruta)+'/Codigo/4 Malla/centroides/c30.npy'
    dir2=str(ruta)+'/Codigo/4 Malla/etiquetas/et_t.npy'
    dir3=str(ruta)+'/Codigo/4 Malla/base de datos/con30_t.npy'
    con30_t=np.load(dir3)
    et_t=np.load(dir2)
    Cen=np.load(dir1)
    conteov=np.zeros((2,len(Cen)))
    
    d_train=acomodar(h_train/255.0)
    datosN_test=quitar_negros(d_train)
    labels_test=KNN(Cen,np.arange(0,len(Cen)),datosN_test)
    conteov[0,:],cluster=contar(len(Cen),labels_test,datosN_test)
    y1_test=np.array([0,1,2,3])
    
    sal_knn,t_knn=KNN_interfaz(con30_t,et_t,conteov,y1_test)
    
    tiempo=(time()-ti)/60
    print("Tiempo de prueba cluster "+str(len(Cen))+" \n "+str(tiempo)+" minutos")

    return sal_knn

def intersecion_interfaz(X1_train,y1_train,X1_test):
    
    
    #X1_train=con40_t
    #y1_train=et_t
    #X1_test=conteov
    
    import numpy as np
    from sklearn.metrics import accuracy_score
    import statistics 
    from time import time,localtime
    
    '''
    actual=localtime()
    print("            prueba interseccion interfaz "+str(actual[3])+":"+str(actual[4]))    
    ti = time()
    '''
    X,Y=X1_train.shape
    Z=len(X1_test)
    interse=np.zeros((Y,X,Z))
    sal=np.zeros(4)
    for c in range(0,Z):
        for a in range(0,X):
            for b in range(0,Y):
                if (X1_train[a,b]<=X1_test[c,b]):
                    interse[b,a,c]=X1_train[a,b]
                else:
                    interse[b,a,c]=X1_test[c,b]
    
    suma_interse=sum(interse)
    maximo_valor=np.zeros(Z)
    etiqueta=np.zeros(Z)
    for c in range(0,Z):
        aux=np.where(suma_interse[:,c] == max(suma_interse[:,c]))[0] 
        #print('posicion: '+str(aux))
        maximo_valor[c]=statistics.mean(aux)
        #print(maximo_valor)
        
        if(maximo_valor[c]>=0 and maximo_valor[c]<int(X/4)):
            etiqueta[c]=0
        if(maximo_valor[c]>int(X/4) and maximo_valor[c]<int(X/4)*2):
            etiqueta[c]=1
        if(maximo_valor[c]>=int(X/4)*2 and maximo_valor[c]<int(X/4)*3):
            etiqueta[c]=2
        if(maximo_valor[c]>=int(X/4)*3 and maximo_valor[c]<int(X/4)*4):
            etiqueta[c]=3
    '''
    print('etiqueta: '+str(etiqueta))
    tiempo=(time()-ti)/60
    print("Tiempo de prueba cluster "+str((Y))+" \n "+str(tiempo)+" minutos")
    '''
    return etiqueta

def prueba_interfaz_intersecion(h_train):
    
    import math
    import numpy as np
    from time import time,localtime
    ruta='D:/Maestria/Tesis 3/'
    
    actual=localtime()
    print("            prueba "+str(actual[3])+":"+str(actual[4]))    
    ti = time()
    
    dir1=str(ruta)+'/Codigo/4 Malla/centroides/c30.npy'
    dir2=str(ruta)+'/Codigo/4 Malla/etiquetas/et_t.npy'
    dir3=str(ruta)+'/Codigo/4 Malla/base de datos/con30_t.npy'
    con40_t=np.load(dir3)
    et_t=np.load(dir2)
    Cen=np.load(dir1)
    conteov=np.zeros((2,len(Cen)))
    
    d_train=acomodar(h_train/255.0)
    #d_train=acomodar(h_train)
    datosN_test=quitar_negros(d_train)
    labels_test=KNN(Cen,np.arange(0,len(Cen)),datosN_test)
    conteov[0,:],cluster=contar(len(Cen),labels_test,datosN_test)
    
    sal_knn=intersecion_interfaz(con40_t,et_t,conteov)
    
    tiempo=(time()-ti)*1000
    print("Tiempo de prueba cluster "+str(len(Cen))+" \n "+str(tiempo)+" mili-segundos")

    return sal_knn

# X1_train: magnitudes de cada una de las flores
# y1_train: etiquetas de las magnitudes
# X1_test: datos de validacion
# y1_test: etiqueta de cada una de las flores de validacion
# sal: exactud entre los datos 
def intersecion(X1_train,y1_train,X1_test,y1_test):
    
    import numpy as np
    from sklearn.metrics import accuracy_score
    import statistics 
    from time import time,localtime
    from sklearn.metrics import confusion_matrix
    
    '''
    X1_train=con20_t
    y1_train=et_t
    X1_test=conv20_t
    y1_test=et_v
    '''
    actual=localtime()
    print("            prueba "+str(actual[3])+":"+str(actual[4]))    
    ti = time()
    
    X,Y=X1_train.shape
    Z=len(X1_test)
    interse=np.zeros((Y,X,Z))
    sal=np.zeros(4)
    for c in range(0,Z):
        for a in range(0,X):
            for b in range(0,Y):
                if (X1_train[a,b]<=X1_test[c,b]):
                    interse[b,a,c]=X1_train[a,b]
                else:
                    interse[b,a,c]=X1_test[c,b]
    
    suma_interse=sum(interse)
    maximo_valor=np.zeros(Z)
    etiqueta=np.zeros(Z)
    for c in range(0,Z):
        aux=np.where(suma_interse[:,c] == max(suma_interse[:,c]))[0] 
        #print(aux)
        maximo_valor[c]=statistics.mean(aux) 
        
        if(maximo_valor[c]>=0 and maximo_valor[c]<int(X/4)):
            etiqueta[c]=0
        if(maximo_valor[c]>=int(X/4) and maximo_valor[c]<int(X/4)*2):
            etiqueta[c]=1
        if(maximo_valor[c]>=int(X/4)*2 and maximo_valor[c]<int(X/4)*3):
            etiqueta[c]=2
        if(maximo_valor[c]>=int(X/4)*3 and maximo_valor[c]<int(X/4)*4):
            etiqueta[c]=3
          
    sal=(accuracy_score(y1_test, etiqueta))
    confusion=confusion_matrix(y1_test, etiqueta)
    '''
    sal[0]=(accuracy_score(y1_test[0:8], etiqueta[0:8]))
    sal[1]=(accuracy_score(y1_test[8:16], etiqueta[8:16]))
    sal[2]=(accuracy_score(y1_test[16:24], etiqueta[16:24]))
    sal[3]=(accuracy_score(y1_test[24:32], etiqueta[24:32]))
    '''
    tiempo=(time()-ti)*1000
    print("Tiempo de prueba cluster "+str((Y))+" \n "+str(tiempo)+" mili-segundos")
    
    return sal, tiempo

# X1_train: magnitudes de cada una de las flores
# y1_train: etiquetas de las magnitudes
# X1_test: datos de validacion
# y1_test: etiqueta de cada una de las flores de validacion
# sal: exactud entre los datos 
def intersecion_prom(X1_train,y1_train,X1_test,y1_test):
    
    import numpy as np
    import math
    from sklearn.metrics import accuracy_score
    import statistics 
    from time import time,localtime
    
    actual=localtime()
    print("            prueba "+str(actual[3])+":"+str(actual[4]))    
    ti = time()
    
    d1,d2=X1_train.shape
    x1=np.zeros((int(d1/4),d2))
    x1[0,:]=np.matrix.round(sum(X1_train[0:24,:])/24)
    x1[1,:]=np.matrix.round(sum(X1_train[24:48,:])/24)
    x1[2,:]=np.matrix.round(sum(X1_train[48:72,:])/24)
    x1[3,:]=np.matrix.round(sum(X1_train[72:96,:])/24)
    
    X,Y=x1.shape
    Z=len(X1_test)
    interse=np.zeros((Y,X,Z))
    sal=np.zeros(4)
    for c in range(0,Z):
        for a in range(0,X):
            for b in range(0,Y):
                if (x1[a,b]<=X1_test[c,b]):
                    interse[b,a,c]=x1[a,b]
                else:
                    interse[b,a,c]=X1_test[c,b]
    
    suma_interse=sum(interse)
    maximo_valor=np.zeros(Z)
    etiqueta=np.zeros(Z)
    for c in range(0,Z):
        aux=np.where(suma_interse[:,c] == max(suma_interse[:,c]))[0] 
        
        maximo_valor[c]=statistics.mean(aux) 
        print(str(aux)+" maximo: "+str(maximo_valor[c]))
        if(maximo_valor[c]==0):
            etiqueta[c]=0
        if(maximo_valor[c]==1):
            etiqueta[c]=1
        if(maximo_valor[c]==2):
            etiqueta[c]=2
        if(maximo_valor[c]==3):
            etiqueta[c]=3
            
    sal[0]=(accuracy_score(y1_test[0:8], etiqueta[0:8]))
    sal[1]=(accuracy_score(y1_test[8:16], etiqueta[8:16]))
    sal[2]=(accuracy_score(y1_test[16:24], etiqueta[16:24]))
    sal[3]=(accuracy_score(y1_test[24:32], etiqueta[24:32]))
    
    
    tiempo=(time()-ti)/60
    print("Tiempo de prueba cluster "+str((Y))+" \n "+str(tiempo)+" minutos")
    
    return sal,tiempo









def prueba_interfaz_ANN(h_train):
    
    import math
    import numpy as np
    from time import time,localtime
    ruta='D:/Maestria/Tesis 3/'
    
    actual=localtime()
    print("            prueba "+str(actual[3])+":"+str(actual[4]))    
    ti = time()
    
    dir1=str(ruta)+'/Codigo/4 Malla/centroides/c50.npy'
    dir2=str(ruta)+'/Codigo/4 Malla/etiquetas/et_t.npy'
    dir3=str(ruta)+'/Codigo/4 Malla/base de datos/con50_t.npy'
    con50_t=np.load(dir3)
    et_t=np.load(dir2)
    Cen=np.load(dir1)
    conteov=np.zeros((2,len(Cen)))
    
    d_train=acomodar(h_train/255.0)
    datosN_test=quitar_negros(d_train)
    labels_test=KNN(Cen,np.arange(0,len(Cen)),datosN_test)
    conteov[0,:],cluster=contar(len(Cen),labels_test,datosN_test)
    
    sal_knn,t_knn=ANN_interfaz(con50_t,et_t,conteov)
    
    tiempo=(time()-ti)*1000
    print("Tiempo de prueba cluster "+str(len(Cen))+" \n "+str(tiempo)+" mili-segundos")

    return sal_knn

def ANN_interfaz(X_train,Y_train,X_prueba):
    
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    import numpy as np
    from time import time,localtime
    
    actual=localtime()
    print("            codigo "+str(actual[3])+":"+str(actual[4])) 
    ti = time()

    alfa_min=1e-2
    
    clf = MLPClassifier(solver='lbfgs', alpha=alfa_min, hidden_layer_sizes=(160,),max_iter=5000)# ,max_iter=5000,random_state=1
    clf.fit(X_train, Y_train)
    y_pred=clf.predict(X_prueba)
        
    print('paso 1')
    
    tiempo=(time()-ti)*1000
    print("Tiempo ANN    "+str(tiempo)+" mili-segundos") 
    
    return y_pred,tiempo


def promedio_RGB():
    
    import numpy as np
    from time import time,localtime
    import statistics
    import cv2 as cv
    import math
    
    bh=320
    bv=120
    ir='D:/Maestria/Tesis 3/imagenes/muestras python/f4/'
   
    c=np.zeros((bh-1,bv-1,3))
    grafico=np.zeros((40,3))
    
    
    for Z in range(1,41):
        image=cv.imread(ir+str(Z)+".tif")#/255.0
        alto,ancho,dimension=image.shape
        sv=ancho/bv
        sh=alto/bh
        
        for D in range(0,dimension):
            for X in range(1,bh):
                for Y in range(1,bv):
                    c[X-1,Y-1,D]=image[math.floor(sh*X),math.floor(sv*Y),D]
            
        d_train=acomodar(c)
        datosN_test=quitar_negros(d_train)
        grafico[Z-1,:]=sum(datosN_test)/len(datosN_test)
    
    return grafico



