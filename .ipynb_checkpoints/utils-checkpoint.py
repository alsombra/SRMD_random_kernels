import torch
import numpy as np
from PIL import Image
from scipy import signal
from scipy.ndimage import convolve
import h5py, os


class Kernels(object):
    def __init__(self, kernels, proj_matrix):
        self.kernels = kernels   
        self.P = proj_matrix

        # kernels.shape == [H, W, C, N], C: no. of channels / N: no. of kernels
        
        #Ex:kernels.shape = (15,15,1,358)
        
        #Ex: P shape = (15,225)
        self.kernels_proj = np.matmul(self.P,
                                      self.kernels.reshape(self.P.shape[-1],
                                                           self.kernels.shape[-1]))
        #Ex: kernels.shape = (225,358)
        
        #Ex: kernels_proj = (15,358)
        
        self.indices = np.array(range(self.kernels.shape[-1])) #Ex: 0,1,...,357
        #agora escolhe random um kernel
        self.randkern = self.RandomKernel(self.kernels, [self.indices]) #len([indices]) = 1 
                                                                        #indices[0] = [0,1,...,358]
        
    def RandomBlur(self, image):
        kern = next(self.randkern) # aumenta o  
        return Image.fromarray(convolve(image, kern, mode='nearest'))

    def ConcatDegraInfo(self, image):
        #Concatena com kernel aleatório
        image = np.asarray(image)   # PIL Image to numpy array
        h, w = list(image.shape[0:2])
        ######
        self.randkern.index =np.random.randint(len(self.randkern.indices[0]))
        ######
        proj_kernl = self.kernels_proj[:, self.randkern.index]  # Caution!!
        n = len(proj_kernl)  # dim. of proj_kernl

        maps = np.ones((h, w, n))
        for i in range(n):
            maps[:, :, i] = proj_kernl[i] * maps[:, :, i]
        image = np.concatenate((image, maps), axis=-1)
        return image

    class RandomKernel(object):
        def __init__(self, kernels, indices):
            self.len = kernels.shape[-1]         #Ex: 358
            self.indices = indices               #Ex: [[0,1...,357]]
            np.random.shuffle(self.indices[0])   #indices[0] = [0,1,...,357] -> shuffle normal 
            self.kernels = kernels[:, :, :, self.indices[0]] #kernels, mas em uma ordem aleatoria
            self.index = 0

        def __iter__(self):
            return self
    
        def __next__(self):
            
            # RANDOM DRAW WITHOUT REPETION/REPLACEMENT
#             if (self.index == self.len): #quando o indice chegar no final, depois de passar tudo (358x)
#                 np.random.shuffle(self.indices[0]) #Embaralha de novo a ordem
#                 self.kernels = self.kernels[:, :, :, self.indices[0]]
#                 self.index = 0 #reseta index
#           
#            n = self.kernels[:, :, :, self.index] #pega um kernel (o da fila - posicao index) 
#            self.index += 1

            # RANDOM PICK (WITH REPLACEMENT)
            self.index =np.random.randint(len(self.indices[0]))
            n = self.kernels[:, :, :, self.index] #pega um kernel (aleatorio - posicao index)
            
            return n #retorna um kernel


def load_kernels(file_path='kernels/', scale_factor=2):
    f = h5py.File(os.path.join(file_path, 'SRMDNFx%d.mat' % scale_factor), 'r')

    directKernel = None
    if scale_factor != 4:
        directKernel = f['net/meta/directKernel'] #shape = (16,1,15,15)
        directKernel = np.array(directKernel).transpose(3, 2, 1, 0) # new shape = (15,15,1,16)

    AtrpGaussianKernels = f['net/meta/AtrpGaussianKernel'] #shape = (342, 1, 15, 15)
    AtrpGaussianKernels = np.array(AtrpGaussianKernels).transpose(3, 2, 1, 0) #new shape =(15, 15, 1, 342)

    P = f['net/meta/P']
    P = np.array(P)
    P = P.T 
    #P.shape = (15,225) 
    
    if directKernel is not None:                  
        K = np.concatenate((directKernel, AtrpGaussianKernels), axis=-1) #concatena no ultimo axis
                                            #K shape = (15,15,1,16) + (15, 15, 1, 342) = (15,15,1,358)
    else:
        K = AtrpGaussianKernels
    return K, P    #Retorna kernels (anisotropicos(e iso?) e diretos) e matriz de projeçao do PCA 


"""The functions below are not used currently"""


def PCA(data, k=2):
    X = torch.from_numpy(data)
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)

    v, w = torch.eig(torch.mm(X, torch.t(X)), eigenvectors=True)
    return torch.mm(w[:k, :], X)


def isogkern(kernlen, std):
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    gkern2d = gkern2d/np.sum(gkern2d)
    return gkern2d


def anisogkern(kernlen, std1, std2, angle):
    gkern1d_1 = signal.gaussian(kernlen, std=std1).reshape(kernlen, 1)
    gkern1d_2 = signal.gaussian(kernlen, std=std2).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d_1, gkern1d_2)
    gkern2d = gkern2d/np.sum(gkern2d)
    return gkern2d
