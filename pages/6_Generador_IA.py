
                                         #Codificadores automaticos
                                            #Todos los tipos

#Hay veces que al trabajar con imagenes, vamos a querer codificar-comprimir-decodificar
#Esto sirve para trabajar con imagenes comprimidas teniendo las mismas caracterisitcas
#Un codificador automatico, es un modelo de red neuronal que minimiza la diferencia entre la entrada y la salida

#La codificacion ---> tiene una entrada y una salida
#compresion ---> Es solo un mensaje (codigo) que se comprime
#La decodificacion ---> tiene entrada comprimida y una salida lista para ocupar

import os
import copy
import skillsnetwork

import numpy as np
from numpy.core.fromnumeric import reshape
import tensorflow as tf
import keras
from keras import layers,Input,Sequential 
from keras.layers import Dense,Flatten,Reshape,Conv2DTranspose,Conv2D
from keras.models import Model


from keras.layers import Conv2D
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

#Algunas funciones utiles

#Ploteo de imagenes
def plot_images(top,bottom,start=0,stop=5,reshape_x=(28,28),reshape_xhat=(28,28)):
    
    '''
    this function plots images from the start index to the stop index from two datasets
    
    '''

    n_samples=stop-start

    for i,img_index in enumerate(range(start,stop)):
        
        # Display original
        ax = plt.subplot(2, n_samples, i + 1)
        plt.imshow(top[img_index].reshape(reshape_x[0], reshape_x[1]), cmap="gray")

        if i==n_samples//2:
            plt.title("original images")

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n_samples, i + 1 + n_samples)
        plt.imshow(bottom[img_index].reshape(reshape_xhat[0], reshape_xhat[1]), cmap="gray")


        if i==n_samples//2:
            plt.title("encoded images")

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
     
#Muestra la historia de una modelo
def graph_history(history, title='Log Loss and Accuracy over iterations'):

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(title)
    N_plots=len(history.history.keys())
    color_list=['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w','bx','rx']
    for i,(key, items) in enumerate(history.history.items()):
        ax = fig.add_subplot(1, N_plots, i+1)
        ax.plot(items,c=color_list[i])
        ax.grid(True)
        ax.set(xlabel='iterations', title=key)
    plt.show()
    
#Añadir ruido a los datos de entrenamiento y prueba
def add_noise(x_train, x_test, noise_factor = 0.3):
    '''
    this function adds random values from a normal distribution as noises to the data 
    
    returns the noisy datasets 
    
    '''
    noise_factor = 0.3
    x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape) 
    x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape) 

    x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.).numpy()
    x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.).numpy()
    
    return x_train_noisy,x_test_noisy

#Plotear el codigo
def plot_code(h, y, numbers=[0,1,2,3,4,5,6,7,8,9], scale=[1,1,1]):
  
    h=h.numpy()
    color_list=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink','darkorange','lime']
    logic_array =np.zeros(len(y), dtype=bool)
    
    fig=plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    for num, color in zip(numbers, color_list):
        logic_array = (y==num)
        plt.scatter(scale[0]*h[logic_array,0],scale[1]*h[logic_array,1],scale[2]*h[logic_array,2],c=color, label=num)
 
    plt.title("3D output of encoder, colored by digits")
    plt.legend(loc=[1.1,0.3])
    plt.show()

def avg(shape, dtype=None):
    grad = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
        ]).reshape((3, 3, 1, 1))/9
    
    assert grad.shape == shape
    return keras.backend.variable(grad, dtype='float32')

a_conv = Conv2D(filters=1,
                       kernel_size=3,
                       kernel_initializer=avg,
                       strides=1,
                       padding='same')

def display_auto(Xiter,n=1,B=1):

    for b in range(B):    
        x = next(Xiter)
    
        plt.imshow(x[1].numpy()[n,:,:,0],cmap="gray")
        plt.title("input")
        plt.show()
        plt.imshow(x[0].numpy()[n,:,:,0],cmap="gray")
        plt.title("output")
        plt.show()

#Descargamos la data mnist y separamos datos de entrenamiento y prueba
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) =keras.datasets.mnist.load_data()


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

###Hay dos formar de hacer codificadores automaticos con API funcional y con subclasificador

###Forma de API funcional

input_img=Input(shape=(784,))

#Creamos un codificador
encoding_dim=36
encoder = Dense(encoding_dim, activation='relu')(input_img)

#Creamos el decodificador
decoder=Dense(784,activation='sigmoid')(encoder)

#Combinamos el codificador y el decodificador con esto
autoencoder =Model(input_img, decoder)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

history=autoencoder.fit(x_train, x_train,
                epochs=25,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

graph_history(history, title='Log Loss and Accuracy over iterations')

#Vemos las predicciones que hace
xhat=autoencoder.predict(x_test)

#Ploteamos las imagenes de entrada y las de salida de la compresion
#Vemos que son bastante parecidas, mantienen su escencia y calidad en gran medida, pero no iguales
plot_images(x_test,xhat,start=0,stop=5)


###Autocodifcadores con Modelo de Subclasificacion

#En particular este metodo sirve porque se puede reutilizar

class Autoencoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim   
        self.encoder = Dense(latent_dim, activation='relu')
        self.decoder = Dense(784, activation='sigmoid')

    def call(self, x):
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

#Algunas configuraciones
#lateng_sim: Es el tamaño de la entrada

#La codificacion viene dada por
#encoder = tf.keras.Sequential([layers.Dense(latent_dim, activation='relu')])

#La decodificacion dada por
#decoder = tf.keras.Sequential([layers.Dense(784, activation='sigmoid')])

#Este es el unico parametro que necesitamos especificar
encoding_dim=36
autoencoder = Autoencoder(encoding_dim)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history=autoencoder.fit(x_train, x_train,epochs=25,batch_size=256,shuffle=True,validation_data=(x_test, x_test))
graph_history(history, title='Log Loss and Accuracy over iterations')

xhat=autoencoder.predict(x_test)

#Mostramos las predicciones, se parecen bastante a las anteriores aunque 1 se ve bastante disminuida
plot_images(x_test,xhat,start=100,stop=105)

h=autoencoder.encoder(x_test)
print(h.shape)

#Esta parte muestra como estan los datos cuando estan codificados
plot_images(x_test, h.numpy(),start=200,stop=205,reshape_x=(28,28),reshape_xhat=(6,6))

#En general API funcional es mas compatible con diferentes modelos


###Ejemplo 1

#Calculamos la perdida entre cada una de las muestras
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

loss=[bce(x, x_s).numpy() for x, x_s in zip (x_test,xhat)]

indexs=np.flip(np.argsort(loss))

#Grafica la informacion que pierde cada imagen
plt.figure(figsize=(18,3))
for i, index in enumerate(indexs[0:10]):

    plt.subplot(1, 10, i+1)
    plt.imshow(x_test[index].reshape(28,28))
    plt.title(f"No.{index}")
    plt.axis("off")
plt.show()


###Eliminacion de ruido

#Podemos entrenar la red con imagenes con ruido primero

#Le agregamos ruido a las imagenes
x_train_noisy,x_test_noisy= add_noise(x_train, x_test,noise_factor = 0.4)

fig1 = plt.figure(figsize=(10,2))
fig1.suptitle("original images")
fig2 = plt.figure(figsize=(10,2))
fig2.suptitle("noisy images")

#Ploteamos las imagenes originales y las con ruido
for i, img_index in enumerate(range(5)):
    ax1 = fig1.add_subplot(1, 5, i+1)
    ax1.imshow(x_train[img_index].reshape((28,28)))
    ax1.axis("off")
    ax2 = fig2.add_subplot(1, 5, i+1)
    ax2.imshow(x_train_noisy[img_index].reshape((28,28)))
    ax2.axis("off")
plt.show()


#Una forma de eliminar ruido tambien con facilidad es aumentando la imagen de tamaño
encoding_dim=2*x_test.shape[1]
autoencoder = Autoencoder(encoding_dim)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history=autoencoder.fit(x_train_noisy , x_train,epochs=25,batch_size=256,shuffle=True,validation_data=(x_test_noisy, x_test))
graph_history(history, title='Log Loss and Accuracy over iterations')
xhat=autoencoder.predict(x_test)

fig1 = plt.figure(figsize=(10,2))
fig1.suptitle("noisy images")
fig2 = plt.figure(figsize=(10,2))
fig2.suptitle("de-noised images")

for i, img_index in enumerate(range(5)):
    ax1 = fig1.add_subplot(1, 5, i+1)
    ax1.imshow(x_test_noisy[img_index].reshape((28,28)))
    ax1.axis("off")
    ax2 = fig2.add_subplot(1, 5, i+1)
    ax2.imshow(xhat[img_index].reshape((28,28)))
    ax2.axis("off")
plt.show()    
    

###Ejemplo 2

#Muestra los valores que contiene los pixeles graficados
autoencoder = Autoencoder(3)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history=autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
h=autoencoder.encoder(x_test)

plot_code(h,y_test)


###Ejemplo 3

#Muestra el resultado de una autocodificicion de las imagenes de moda, las vuelve como objeto bastante borrosa
(x_train, y_train), (x_test,y_test) = keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print(x_train.shape)

x_temp=layers.Flatten()(x_train)
x_temp_test=layers.Flatten()(x_test)
print(x_temp.shape, x_temp_test.shape)

encoding_dim=3
autoencoder = Autoencoder(encoding_dim)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history=autoencoder.fit(x_temp,x_temp,epochs=25,batch_size=256,shuffle=True,validation_data=(x_temp_test,x_temp_test))

xhat=autoencoder.predict(x_temp_test)
plot_images(x_test,xhat,start=0,stop=5)



###AUTOCODIFICADORES PROFUNDOS (DEEP AUTOCODERS)

#En el ejemplo anterior se vio que los autocidicadores no siempre funncionan bien, de hecho el ultimo ejemplo es casi
#irreconocible el elemento

#Por esta razon se pueden entrenar modelos para que puedan obtener las mejores caracterisiticas

#Una de las cosas mas importantes es que las deep autocoders NO TIENEN LABELS

class Deep_Autoencoder (Model):
    def __init__(self, latent_dim_1, latent_dim_2):
        super(Deep_Autoencoder, self).__init__()
        self.latent_dim_1= latent_dim_1  
        self.latent_dim_1= latent_dim_2 
        self.encoder = Sequential([layers.Flatten(),Dense(latent_dim_1, activation='relu'),Dense(latent_dim_2, activation='relu')])
        self.decoder = tf.keras.Sequential([Dense(latent_dim_1, activation='relu'), Dense(784, activation='sigmoid'), Reshape((28, 28))])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

#De forma de API:
#encoder = Sequential([layers.Flatten(),Dense(latent_dim_1, activation='relu'),Dense(latent_dim_2, activation='relu')])
#decoder = tf.keras.Sequential([ Dense(latent_dim_1, activation='relu'),Dense(784, activation='sigmoid'),Reshape((28, 28))])

latent_dim_1 =128
latent_dim_2=3
deep_autoencoder=Deep_Autoencoder(latent_dim_1=latent_dim_1,latent_dim_2=latent_dim_2)


deep_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history=deep_autoencoder.fit(x_train,x_train,epochs=50,batch_size=256,shuffle=True,validation_data=(x_test,x_test))

xhat=deep_autoencoder.predict(x_test)
plot_images(x_test,xhat,start=0,stop=5)

###Ejemplo de deep autocodificadores

#Descargamos la data

img_height=50
img_width=50
batch_size=100
data_dir_face=os.path.join(os.getcwd(), 'face_data')

Xface = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_face,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
    color_mode="grayscale")

X_face_copy=copy.copy(Xface)

def change_inputs(images, labels):
  
    return images, images

X_face_1=Xface.map(change_inputs)

X_iter=iter(X_face_1)

images1, images2 = next(X_iter)

print("images1 shape {}, and images2 {}".format(images1.shape, images2.shape))

display_auto(X_iter,n=1,B=1)

normalization_layer = tf.keras.layers.Rescaling(1./255)

#def blur_image(images, labels):
    
#    x = normalization_layer(images)
#    x_b=a_conv(x)
#    return x_b, x


Xface=Xface.map(lambda images, labels: (a_conv(normalization_layer(images)), labels))

Xiter=iter(Xface)
display_auto(Xiter,n=1,B=1)


###Autocodificadores convolucionales

#La diferencias con los deep autocoders es que los autodificadores convolucionales pueden, elegir mejor las caracterisitcas
#vimos que a pesar de que mejoro en el ejemplo anterior no es tan parecida a la imagen

class CNN_Autoencoder(Model):
    def __init__(self):
        super(CNN_Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(50, 50, 1)),
            Conv2D(16, (3, 3), activation='relu', padding='same', strides=1),
            Conv2D(8, (3, 3), activation='relu', padding='same', strides=1)])

        self.decoder = tf.keras.Sequential([
            Conv2DTranspose(8, kernel_size=3, strides=1, activation='relu', padding='same'),
            Conv2DTranspose(16, kernel_size=3, strides=1, activation='relu', padding='same'),
            Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

#En forma de API funcional:

#encoder = tf.keras.Sequential([layers.Input(shape=(50, 50, 1)),Conv2D(16, (3, 3), activation='relu', padding='same', strides=1),Conv2D(8, (3, 3), activation='relu', padding='same', strides=1)])
#decoder = tf.keras.Sequential([Conv2DTranspose(8, kernel_size=3, strides=1, activation='relu', padding='same'),Conv2DTranspose(16, kernel_size=3, strides=1, activation='relu', padding='same'),Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

cnn_autoencoder_face=CNN_Autoencoder()

cnn_autoencoder_face.compile(optimizer='adam',  loss='mse')
history=cnn_autoencoder_face.fit(Xface,epochs=10)

graph_history(history, title='Log Loss and Accuracy over iterations')

x = next(Xiter)
Xhat=cnn_autoencoder_face.predict(x[0])

for image_b, image_db in zip(x[0].numpy()[0:5,:,:,0],Xhat[0:5,:,:,0]):
    plt.imshow(image_b, cmap="gray")
    plt.title("blurred image")
    plt.show()
    plt.imshow(image_db, cmap="gray")
    plt.title("de-blurred image")
    plt.show()


