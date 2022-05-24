# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 17:50:04 2020

@author: Cristian Gonzalez
"""


import malla
import numpy as np
import math
from time import time,localtime
ruta='D:/Maestria/Tesis 3/'

tci = time()
actual=localtime()
print("\n\n            inicio del programa "+str(actual[3])+":"+str(actual[4]))

f=np.zeros((4))
f[0]=malla.read(str(ruta)+'/imagenes/muestras python/f1/*.tif')
h1=malla.matriz(320,120,math.floor(f[0]*0.6),str(ruta)+'/imagenes/muestras python/f1/')

f[1]=malla.read(str(ruta)+'/imagenes/muestras python/f2/*.tif')
h2=malla.matriz(320,120,math.floor(f[1]*0.6),str(ruta)+'/imagenes/muestras python/f2/')

f[2]=malla.read(str(ruta)+'/imagenes/muestras python/f3/*.tif')
h3=malla.matriz(320,120,math.floor(f[2]*0.6),str(ruta)+'/imagenes/muestras python/f3/')

f[3]=malla.read(str(ruta)+'/imagenes/muestras python/f4/*.tif')
h4=malla.matriz(320,120,math.floor(f[3]*0.6),str(ruta)+'/imagenes/muestras python/f4/')

print("tiempo creacion de datos de entrenamiento "+str((time()-tci)/60)+" minutos")

print("convirtiendo de 3 a 2 dimensiones")
d=np.zeros((h1.shape[0]*h1.shape[1],3,4))
d[:,:,0]=malla.acomodar(h1)
d[:,:,1]=malla.acomodar(h2)
d[:,:,2]=malla.acomodar(h3)
d[:,:,3]=malla.acomodar(h4)


print("apilando las familias de flores")
datos=np.concatenate((d[:,:,0],d[:,:,1],d[:,:,2],d[:,:,3]))

print("quitando el color negro")
datosN=malla.quitar_negros(datos)


"""
         bag of words
"""
t_word=np.zeros((10))
d10,c10,labels10,t_word[0]=malla.codigo(datosN,10)
d20,c20,labels20,t_word[1]=malla.codigo(datosN,20)
d30,c30,labels30,t_word[2]=malla.codigo(datosN,30)
d40,c40,labels40,t_word[3]=malla.codigo(datosN,40)
d50,c50,labels50,t_word[4]=malla.codigo(datosN,50)
d60,c60,labels60,t_word[5]=malla.codigo(datosN,60)
d70,c70,labels70,t_word[6]=malla.codigo(datosN,70)
d80,c80,labels80,t_word[7]=malla.codigo(datosN,80)
d90,c90,labels90,t_word[8]=malla.codigo(datosN,90)
d100,c100,labels100,t_word[9]=malla.codigo(datosN,100)

np.save(str(ruta)+'/Codigo/4 Malla/histogramas/d10',d10)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/d20',d20)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/d30',d30)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/d40',d40)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/d50',d50)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/d60',d60)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/d70',d70)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/d80',d80)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/d90',d90)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/d100',d100)
np.save(str(ruta)+'/Codigo/4 Malla/centroides/c10',c10)
np.save(str(ruta)+'/Codigo/4 Malla/centroides/c20',c20)
np.save(str(ruta)+'/Codigo/4 Malla/centroides/c30',c30)
np.save(str(ruta)+'/Codigo/4 Malla/centroides/c40',c40)
np.save(str(ruta)+'/Codigo/4 Malla/centroides/c50',c50)
np.save(str(ruta)+'/Codigo/4 Malla/centroides/c60',c60)
np.save(str(ruta)+'/Codigo/4 Malla/centroides/c70',c70)
np.save(str(ruta)+'/Codigo/4 Malla/centroides/c80',c80)
np.save(str(ruta)+'/Codigo/4 Malla/centroides/c90',c90)
np.save(str(ruta)+'/Codigo/4 Malla/centroides/c100',c100)
np.save(str(ruta)+'/Codigo/4 Malla/etiquetas/labels10',labels10)
np.save(str(ruta)+'/Codigo/4 Malla/etiquetas/labels20',labels20)
np.save(str(ruta)+'/Codigo/4 Malla/etiquetas/labels30',labels30)
np.save(str(ruta)+'/Codigo/4 Malla/etiquetas/labels40',labels40)
np.save(str(ruta)+'/Codigo/4 Malla/etiquetas/labels50',labels50)
np.save(str(ruta)+'/Codigo/4 Malla/etiquetas/labels60',labels60)
np.save(str(ruta)+'/Codigo/4 Malla/etiquetas/labels70',labels70)
np.save(str(ruta)+'/Codigo/4 Malla/etiquetas/labels80',labels80)
np.save(str(ruta)+'/Codigo/4 Malla/etiquetas/labels90',labels90)
np.save(str(ruta)+'/Codigo/4 Malla/etiquetas/labels100',labels100)
np.save(str(ruta)+'/Codigo/4 Malla/tiempos/t_word',t_word)


"""
    creando histograma de los datos de entrenamiento validacion y prueba
"""

t_en=np.zeros((10))
et_t,con10_t,t_en[0]=malla.entrenamiento(f[0],f[1],f[2],f[3],labels10,10)
et_t,con20_t,t_en[1]=malla.entrenamiento(f[0],f[1],f[2],f[3],labels20,20)
et_t,con30_t,t_en[2]=malla.entrenamiento(f[0],f[1],f[2],f[3],labels30,30)
et_t,con40_t,t_en[3]=malla.entrenamiento(f[0],f[1],f[2],f[3],labels40,40)
et_t,con50_t,t_en[4]=malla.entrenamiento(f[0],f[1],f[2],f[3],labels50,50)
et_t,con60_t,t_en[5]=malla.entrenamiento(f[0],f[1],f[2],f[3],labels60,60)
et_t,con70_t,t_en[6]=malla.entrenamiento(f[0],f[1],f[2],f[3],labels70,70)
et_t,con80_t,t_en[7]=malla.entrenamiento(f[0],f[1],f[2],f[3],labels80,80)
et_t,con90_t,t_en[8]=malla.entrenamiento(f[0],f[1],f[2],f[3],labels90,90)
et_t,con100_t,t_en[9]=malla.entrenamiento(f[0],f[1],f[2],f[3],labels100,100)

np.save(str(ruta)+'/Codigo/4 Malla/etiquetas/et_t',et_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/con10_t',con10_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/con20_t',con20_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/con30_t',con30_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/con40_t',con40_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/con50_t',con50_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/con60_t',con60_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/con70_t',con70_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/con80_t',con80_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/con90_t',con90_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/con100_t',con100_t)
np.save(str(ruta)+'/Codigo/4 Malla/tiempos/t_en',t_en)


t_v=np.zeros((10))
et_v,conv10_t,t_v[0]=malla.validacion(f[0],f[1],f[2],f[3],c10)
et_v,conv20_t,t_v[1]=malla.validacion(f[0],f[1],f[2],f[3],c20)
et_v,conv30_t,t_v[2]=malla.validacion(f[0],f[1],f[2],f[3],c30)
et_v,conv40_t,t_v[3]=malla.validacion(f[0],f[1],f[2],f[3],c40)
et_v,conv50_t,t_v[4]=malla.validacion(f[0],f[1],f[2],f[3],c50)
et_v,conv60_t,t_v[5]=malla.validacion(f[0],f[1],f[2],f[3],c60)
et_v,conv70_t,t_v[6]=malla.validacion(f[0],f[1],f[2],f[3],c70)
et_v,conv80_t,t_v[7]=malla.validacion(f[0],f[1],f[2],f[3],c80)
et_v,conv90_t,t_v[8]=malla.validacion(f[0],f[1],f[2],f[3],c90)
et_v,conv100_t,t_v[9]=malla.validacion(f[0],f[1],f[2],f[3],c100)

np.save(str(ruta)+'/Codigo/4 Malla/etiquetas/et_v',et_v)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/conv10_t',conv10_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/conv20_t',conv20_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/conv30_t',conv30_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/conv40_t',conv40_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/conv50_t',conv50_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/conv60_t',conv60_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/conv70_t',conv70_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/conv80_t',conv80_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/conv90_t',conv90_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/conv100_t',conv100_t)
np.save(str(ruta)+'/Codigo/4 Malla/tiempos/t_en',t_en)


t_p=np.zeros((10))
et_p,conp10_t,t_p[0]=malla.prueba(f[0],f[1],f[2],f[3],c10)
et_p,conp20_t,t_p[1]=malla.prueba(f[0],f[1],f[2],f[3],c20)
et_p,conp30_t,t_p[2]=malla.prueba(f[0],f[1],f[2],f[3],c30)
et_p,conp40_t,t_p[3]=malla.prueba(f[0],f[1],f[2],f[3],c40)
et_p,conp50_t,t_p[4]=malla.prueba(f[0],f[1],f[2],f[3],c50)
et_p,conp60_t,t_p[5]=malla.prueba(f[0],f[1],f[2],f[3],c60)
et_p,conp70_t,t_p[6]=malla.prueba(f[0],f[1],f[2],f[3],c70)
et_p,conp80_t,t_p[7]=malla.prueba(f[0],f[1],f[2],f[3],c80)
et_p,conp90_t,t_p[8]=malla.prueba(f[0],f[1],f[2],f[3],c90)
et_p,conp100_t,t_p[9]=malla.prueba(f[0],f[1],f[2],f[3],c100)

np.save(str(ruta)+'/Codigo/4 Malla/etiquetas/et_p',et_p)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/conp10_t',conp10_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/conp20_t',conp20_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/conp30_t',conp30_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/conp40_t',conp40_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/conp50_t',conp50_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/conp60_t',conp60_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/conp70_t',conp70_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/conp80_t',conp80_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/conp90_t',conp90_t)
np.save(str(ruta)+'/Codigo/4 Malla/histogramas/conp100_t',conp100_t)
np.save(str(ruta)+'/Codigo/4 Malla/tiempos/t_p',t_en)

""""
====================================================
"""

"""
            Clasificador KNN
"""
t_knn=np.zeros((10))
sal_knn=np.zeros((29,10))
sal_knn[:,0],t_knn[0]=malla.KNN2(con10_t,et_t,conp10_t,et_p)
sal_knn[:,1],t_knn[1]=malla.KNN2(con20_t,et_t,conp20_t,et_p)
sal_knn[:,2],t_knn[2]=malla.KNN2(con30_t,et_t,conp30_t,et_p)
sal_knn[:,3],t_knn[3]=malla.KNN2(con40_t,et_t,conp40_t,et_p)
sal_knn[:,4],t_knn[4]=malla.KNN2(con50_t,et_t,conp50_t,et_p)
sal_knn[:,5],t_knn[5]=malla.KNN2(con60_t,et_t,conp60_t,et_p)
sal_knn[:,6],t_knn[6]=malla.KNN2(con70_t,et_t,conp70_t,et_p)
sal_knn[:,7],t_knn[7]=malla.KNN2(con80_t,et_t,conp80_t,et_p)
sal_knn[:,8],t_knn[8]=malla.KNN2(con90_t,et_t,conp90_t,et_p)
sal_knn[:,9],t_knn[9]=malla.KNN2(con100_t,et_t,conp100_t,et_p)

np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_knn',sal_knn)
np.save(str(ruta)+'/Codigo/4 Malla/tiempos/t_knn',t_knn)

"""
            Clasificador SVM
"""
t_svm1=np.zeros((10))
sal_svm1=np.zeros((10))
sal_svm1[0],t_svm1[0]=malla.svm1(con10_t,et_t,conp10_t,et_p)
sal_svm1[1],t_svm1[1]=malla.svm1(con20_t,et_t,conp20_t,et_p)
sal_svm1[2],t_svm1[2]=malla.svm1(con30_t,et_t,conp30_t,et_p)
sal_svm1[3],t_svm1[3]=malla.svm1(con40_t,et_t,conp40_t,et_p)
sal_svm1[4],t_svm1[4]=malla.svm1(con50_t,et_t,conp50_t,et_p)
sal_svm1[5],t_svm1[5]=malla.svm1(con60_t,et_t,conp60_t,et_p)
sal_svm1[6],t_svm1[6]=malla.svm1(con70_t,et_t,conp70_t,et_p)
sal_svm1[7],t_svm1[7]=malla.svm1(con80_t,et_t,conp80_t,et_p)
sal_svm1[8],t_svm1[8]=malla.svm1(con90_t,et_t,conp90_t,et_p)
sal_svm1[9],t_svm1[9]=malla.svm1(con100_t,et_t,conp100_t,et_p)

np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_svm1',sal_svm1)
np.save(str(ruta)+'/Codigo/4 Malla/tiempos/t_svm1',t_svm1)

t_svm2=np.zeros((10))
sal_svm2=np.zeros((10))
sal_svm2[0],t_svm2[0]=malla.svm2(con10_t,et_t,conp10_t,et_p)
sal_svm2[1],t_svm2[1]=malla.svm2(con20_t,et_t,conp20_t,et_p)
sal_svm2[2],t_svm2[2]=malla.svm2(con30_t,et_t,conp30_t,et_p)
sal_svm2[3],t_svm2[3]=malla.svm2(con40_t,et_t,conp40_t,et_p)
sal_svm2[4],t_svm2[4]=malla.svm2(con50_t,et_t,conp50_t,et_p)
sal_svm2[5],t_svm2[5]=malla.svm2(con60_t,et_t,conp60_t,et_p)
sal_svm2[6],t_svm2[6]=malla.svm2(con70_t,et_t,conp70_t,et_p)
sal_svm2[7],t_svm2[7]=malla.svm2(con80_t,et_t,conp80_t,et_p)
sal_svm2[8],t_svm2[8]=malla.svm2(con90_t,et_t,conp90_t,et_p)
sal_svm2[9],t_svm2[9]=malla.svm2(con100_t,et_t,conp100_t,et_p)

np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_svm2',sal_svm2)
np.save(str(ruta)+'/Codigo/4 Malla/tiempos/t_svm2',t_svm2)

t_svm3=np.zeros((10))
sal_svm3=np.zeros((10))
sal_svm3[0],t_svm3[0]=malla.svm3(con10_t,et_t,conp10_t,et_p)
sal_svm3[1],t_svm3[1]=malla.svm3(con20_t,et_t,conp20_t,et_p)
sal_svm3[2],t_svm3[2]=malla.svm3(con30_t,et_t,conp30_t,et_p)
sal_svm3[3],t_svm3[3]=malla.svm3(con40_t,et_t,conp40_t,et_p)
sal_svm3[4],t_svm3[4]=malla.svm3(con50_t,et_t,conp50_t,et_p)
sal_svm3[5],t_svm3[5]=malla.svm3(con60_t,et_t,conp60_t,et_p)
sal_svm3[6],t_svm3[6]=malla.svm3(con70_t,et_t,conp70_t,et_p)
sal_svm3[7],t_svm3[7]=malla.svm3(con80_t,et_t,conp80_t,et_p)
sal_svm3[8],t_svm3[8]=malla.svm3(con90_t,et_t,conp90_t,et_p)
sal_svm3[9],t_svm3[9]=malla.svm3(con100_t,et_t,conp100_t,et_p)

np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_svm3',sal_svm3)
np.save(str(ruta)+'/Codigo/4 Malla/tiempos/t_svm3',t_svm3)

grado=10
t_svm4=np.zeros((10))
sal_svm4=np.zeros((10,grado))
sal_svm4[0],t_svm4[0]=malla.svm4(con10_t,et_t,conp10_t,et_p,grado)
sal_svm4[1],t_svm4[1]=malla.svm4(con20_t,et_t,conp20_t,et_p,grado)
sal_svm4[2],t_svm4[2]=malla.svm4(con30_t,et_t,conp30_t,et_p,grado)
sal_svm4[3],t_svm4[3]=malla.svm4(con40_t,et_t,conp40_t,et_p,grado)
sal_svm4[4],t_svm4[4]=malla.svm4(con50_t,et_t,conp50_t,et_p,grado)
sal_svm4[5],t_svm4[5]=malla.svm4(con60_t,et_t,conp60_t,et_p,grado)
sal_svm4[6],t_svm4[6]=malla.svm4(con70_t,et_t,conp70_t,et_p,grado)
sal_svm4[7],t_svm4[7]=malla.svm4(con80_t,et_t,conp80_t,et_p,grado)
sal_svm4[8],t_svm4[8]=malla.svm4(con90_t,et_t,conp90_t,et_p,grado)
sal_svm4[9],t_svm4[9]=malla.svm4(con100_t,et_t,conp100_t,et_p,grado)

np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_svm4',sal_svm4)
np.save(str(ruta)+'/Codigo/4 Malla/tiempos/t_svm4',t_svm4)

"""
            Clasificador ANN
"""
neuronas_max=200
t_ann=np.zeros((10))
sal_ANN_10_CAPA1,sal_ANN_10_CAPA2,sal_ANN_10_CAPA3,sal_ANN_10_CAPA4,sal_ANN_10_CAPA5,t_ann[0]=malla.ANN(con10_t,et_t,conp10_t,et_p,neuronas_max)
sal_ANN_20_CAPA1,sal_ANN_20_CAPA2,sal_ANN_20_CAPA3,sal_ANN_20_CAPA4,sal_ANN_20_CAPA5,t_ann[1]=malla.ANN(con20_t,et_t,conp20_t,et_p,neuronas_max)
sal_ANN_30_CAPA1,sal_ANN_30_CAPA2,sal_ANN_30_CAPA3,sal_ANN_30_CAPA4,sal_ANN_30_CAPA5,t_ann[2]=malla.ANN(con30_t,et_t,conp30_t,et_p,neuronas_max)
sal_ANN_40_CAPA1,sal_ANN_40_CAPA2,sal_ANN_40_CAPA3,sal_ANN_40_CAPA4,sal_ANN_40_CAPA5,t_ann[3]=malla.ANN(con40_t,et_t,conp40_t,et_p,neuronas_max)
sal_ANN_50_CAPA1,sal_ANN_50_CAPA2,sal_ANN_50_CAPA3,sal_ANN_50_CAPA4,sal_ANN_50_CAPA5,t_ann[4]=malla.ANN(con50_t,et_t,conp50_t,et_p,neuronas_max)
sal_ANN_60_CAPA1,sal_ANN_60_CAPA2,sal_ANN_60_CAPA3,sal_ANN_60_CAPA4,sal_ANN_60_CAPA5,t_ann[5]=malla.ANN(con60_t,et_t,conp60_t,et_p,neuronas_max)
sal_ANN_70_CAPA1,sal_ANN_70_CAPA2,sal_ANN_70_CAPA3,sal_ANN_70_CAPA4,sal_ANN_70_CAPA5,t_ann[6]=malla.ANN(con70_t,et_t,conp70_t,et_p,neuronas_max)
sal_ANN_80_CAPA1,sal_ANN_80_CAPA2,sal_ANN_80_CAPA3,sal_ANN_80_CAPA4,sal_ANN_80_CAPA5,t_ann[7]=malla.ANN(con80_t,et_t,conp80_t,et_p,neuronas_max)
sal_ANN_90_CAPA1,sal_ANN_90_CAPA2,sal_ANN_90_CAPA3,sal_ANN_90_CAPA4,sal_ANN_90_CAPA5,t_ann[8]=malla.ANN(con90_t,et_t,conp90_t,et_p,neuronas_max)
sal_ANN_100_CAPA1,sal_ANN_100_CAPA2,sal_ANN_100_CAPA3,sal_ANN_100_CAPA4,sal_ANN_100_CAPA5,t_ann[9]=malla.ANN(con100_t,et_t,conp100_t,et_p,neuronas_max)

sal_ANN_50_CAPA1,t_ann[4]= malla.ANN2(con50_t,et_t,conp50_t,et_p,neuronas_max)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_10_CAPA1',sal_ANN_10_CAPA1)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_10_CAPA2',sal_ANN_10_CAPA2)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_10_CAPA3',sal_ANN_10_CAPA3)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_10_CAPA4',sal_ANN_10_CAPA4)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_10_CAPA5',sal_ANN_10_CAPA5)

np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_20_CAPA1',sal_ANN_20_CAPA1)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_20_CAPA2',sal_ANN_20_CAPA2)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_20_CAPA3',sal_ANN_20_CAPA3)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_20_CAPA4',sal_ANN_20_CAPA4)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_20_CAPA5',sal_ANN_20_CAPA5)

np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_30_CAPA1',sal_ANN_30_CAPA1)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_30_CAPA2',sal_ANN_30_CAPA2)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_30_CAPA3',sal_ANN_30_CAPA3)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_30_CAPA4',sal_ANN_30_CAPA4)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_30_CAPA5',sal_ANN_30_CAPA5)

np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_40_CAPA1',sal_ANN_40_CAPA1)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_40_CAPA2',sal_ANN_40_CAPA2)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_40_CAPA3',sal_ANN_40_CAPA3)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_40_CAPA4',sal_ANN_40_CAPA4)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_40_CAPA5',sal_ANN_40_CAPA5)

np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_50_CAPA1',sal_ANN_50_CAPA1)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_50_CAPA2',sal_ANN_50_CAPA2)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_50_CAPA3',sal_ANN_50_CAPA3)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_50_CAPA4',sal_ANN_50_CAPA4)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_50_CAPA5',sal_ANN_50_CAPA5)

np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_60_CAPA1',sal_ANN_60_CAPA1)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_60_CAPA2',sal_ANN_60_CAPA2)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_60_CAPA3',sal_ANN_60_CAPA3)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_60_CAPA4',sal_ANN_60_CAPA4)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_60_CAPA5',sal_ANN_60_CAPA5)

np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_70_CAPA1',sal_ANN_70_CAPA1)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_70_CAPA2',sal_ANN_70_CAPA2)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_70_CAPA3',sal_ANN_70_CAPA3)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_70_CAPA4',sal_ANN_70_CAPA4)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_70_CAPA5',sal_ANN_70_CAPA5)

np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_80_CAPA1',sal_ANN_80_CAPA1)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_80_CAPA2',sal_ANN_80_CAPA2)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_80_CAPA3',sal_ANN_80_CAPA3)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_80_CAPA4',sal_ANN_80_CAPA4)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_80_CAPA5',sal_ANN_80_CAPA5)

np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_90_CAPA1',sal_ANN_90_CAPA1)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_90_CAPA2',sal_ANN_90_CAPA2)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_90_CAPA3',sal_ANN_90_CAPA3)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_90_CAPA4',sal_ANN_90_CAPA4)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_90_CAPA5',sal_ANN_90_CAPA5)

np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_100_CAPA1',sal_ANN_100_CAPA1)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_100_CAPA2',sal_ANN_100_CAPA2)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_100_CAPA3',sal_ANN_100_CAPA3)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_100_CAPA4',sal_ANN_100_CAPA4)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_100_CAPA5',sal_ANN_100_CAPA5)

np.save(str(ruta)+'/Codigo/4 Malla/tiempos/t_ann',t_ann)

"""
            Clasificador intersecion
"""
t_intersecion=np.zeros((10))
sal_intersecion=np.zeros((4,10))
sal_intersecion[:,0],t_intersecion[0]=malla.intersecion(con10_t,et_t,conp10_t,et_p)
sal_intersecion[:,1],t_intersecion[1]=malla.intersecion(con20_t,et_t,conp20_t,et_p)
sal_intersecion[:,2],t_intersecion[2]=malla.intersecion(con30_t,et_t,conp30_t,et_p)
sal_intersecion[:,3],t_intersecion[3]=malla.intersecion(con40_t,et_t,conp40_t,et_p)
sal_intersecion[:,4],t_intersecion[4]=malla.intersecion(con50_t,et_t,conp50_t,et_p)
sal_intersecion[:,5],t_intersecion[5]=malla.intersecion(con60_t,et_t,conp60_t,et_p)
sal_intersecion[:,6],t_intersecion[6]=malla.intersecion(con70_t,et_t,conp70_t,et_p)
sal_intersecion[:,7],t_intersecion[7]=malla.intersecion(con80_t,et_t,conp80_t,et_p)
sal_intersecion[:,8],t_intersecion[8]=malla.intersecion(con90_t,et_t,conp90_t,et_p)
sal_intersecion[:,9],t_intersecion[9]=malla.intersecion(con100_t,et_t,conp100_t,et_p)

np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_intersecion',sal_intersecion)
np.save(str(ruta)+'/Codigo/4 Malla/tiempos/t_intersecion',t_intersecion)

"""
            Clasificador intersecion promedio
"""
t_intersecion_prom=np.zeros((10))
sal_intersecion_prom=np.zeros((4,10))
sal_intersecion_prom[:,0],t_intersecion_prom[0]=malla.intersecion_prom(con10_t,et_t,conp10_t,et_p)
sal_intersecion_prom[:,1],t_intersecion_prom[1]=malla.intersecion_prom(con20_t,et_t,conp20_t,et_p)
sal_intersecion_prom[:,2],t_intersecion_prom[2]=malla.intersecion_prom(con30_t,et_t,conp30_t,et_p)
sal_intersecion_prom[:,3],t_intersecion_prom[3]=malla.intersecion_prom(con40_t,et_t,conp40_t,et_p)
sal_intersecion_prom[:,4],t_intersecion_prom[4]=malla.intersecion_prom(con50_t,et_t,conp50_t,et_p)
sal_intersecion_prom[:,5],t_intersecion_prom[5]=malla.intersecion_prom(con60_t,et_t,conp60_t,et_p)
sal_intersecion_prom[:,6],t_intersecion_prom[6]=malla.intersecion_prom(con70_t,et_t,conp70_t,et_p)
sal_intersecion_prom[:,7],t_intersecion_prom[7]=malla.intersecion_prom(con80_t,et_t,conp80_t,et_p)
sal_intersecion_prom[:,8],t_intersecion_prom[8]=malla.intersecion_prom(con90_t,et_t,conp90_t,et_p)
sal_intersecion_prom[:,9],t_intersecion_prom[9]=malla.intersecion_prom(con100_t,et_t,conp100_t,et_p)

np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_intersecion_prom',sal_intersecion_prom)
np.save(str(ruta)+'/Codigo/4 Malla/tiempos/t_intersecion_prom',t_intersecion_prom)


'''
   ==================         DATOS DE VALIDACION
'''

"""
            Clasificador KNN
"""
t_knn_v=np.zeros((10))
sal_knn_v=np.zeros((29,10))
confusion_v1=np.zeros((4,4,30))
confusion_v2=np.zeros((4,4,30))
confusion_v3=np.zeros((4,4,30))
sal_knn_v[:,1],t_knn_v[1],confusion_v1=malla.KNN2(con20_t,et_t,conv20_t,et_v)
sal_knn_v[:,2],t_knn_v[2],confusion_v2=malla.KNN2(con30_t,et_t,conv30_t,et_v)
sal_knn_v[:,3],t_knn_v[3],confusion_v3=malla.KNN2(con40_t,et_t,conv40_t,et_v)

np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_knn_v',sal_knn_v)
np.save(str(ruta)+'/Codigo/4 Malla/tiempos/t_knn_v',t_knn_v)

"""
            Clasificador SVM
"""
t_svm1_v=np.zeros((10))
sal_svm1_v=np.zeros((10))
sal_svm1_v[2],t_svm1_v[2]=malla.svm1(con30_t,et_t,conv30_t,et_v)
sal_svm1_v[3],t_svm1_v[3]=malla.svm1(con40_t,et_t,conv40_t,et_v)

np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_svm1_v',sal_svm1_v)
np.save(str(ruta)+'/Codigo/4 Malla/tiempos/t_svm1_v',t_svm1_v)



grado=10
t_svm4_v=np.zeros((10))
sal_svm4_v=np.zeros((10,grado))
sal_svm4_v[2],t_svm4_v[2]=malla.svm4(con30_t,et_t,conv30_t,et_v,grado)
sal_svm4_v[3],t_svm4_v[3]=malla.svm4(con40_t,et_t,conv40_t,et_v,grado)

np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_svm4_v',sal_svm4_v)
np.save(str(ruta)+'/Codigo/4 Malla/tiempos/t_svm4_v',t_svm4_v)

"""
            Clasificador ANN
"""
neuronas_max=200
t_ann_v=np.zeros((10))
sal_ANN_30_CAPA1_v,sal_ANN_30_CAPA2_v,sal_ANN_30_CAPA3_v,sal_ANN_30_CAPA4_v,sal_ANN_30_CAPA5_v,t_ann_v[2]=malla.ANN(con30_t,et_t,conv30_t,et_v,neuronas_max)


np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_30_CAPA1_v',sal_ANN_30_CAPA1_v)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_30_CAPA2_v',sal_ANN_30_CAPA2_v)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_30_CAPA3_v',sal_ANN_30_CAPA3_v)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_30_CAPA4_v',sal_ANN_30_CAPA4_v)
np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_ANN_30_CAPA5_v',sal_ANN_30_CAPA5_v)


np.save(str(ruta)+'/Codigo/4 Malla/tiempos/t_ann_v',t_ann_v)

"""
            Clasificador intersecion
"""
t_intersecion_v=np.zeros((10))
sal_intersecion_v=np.zeros((4,10))
sal_intersecion_v[:,2],t_intersecion_v[2]=malla.intersecion(con30_t,et_t,conv30_t,et_v)
sal_intersecion_v[:,6],t_intersecion_v[6]=malla.intersecion(con70_t,et_t,conv70_t,et_v)

np.save(str(ruta)+'/Codigo/4 Malla/resultados/sal_intersecion_v',sal_intersecion_v)
np.save(str(ruta)+'/Codigo/4 Malla/tiempos/t_intersecion_v',t_intersecion_v)






print("\n\n Duracion del codigo "+str((time()-tci)/60)+" minutos")





