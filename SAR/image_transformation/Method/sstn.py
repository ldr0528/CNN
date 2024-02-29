#import cv2
import numpy as np

#Data = cv2.imread('662_9111_129_32_vh.jpg',0)
#Data = cv2.resize(Data,[256,256])/256
#s = Data[0,:]
def SSTN(s,gamma,sigma):
#    gamma = 0.0100;
#    sigma = 0.055;
    
    n = len(s)
    
    ft = np.linspace(1,int(n/2),int(n/2))
    bt = np.linspace(1,n,n)
    
    nb = len(bt)
    neta = len(ft)
    sz = np.zeros([n,1])
    
    sleft = np.flipud(np.conj(sz[1:int(n/2+1)]))
    sright = np.flipud(sz[(len(sz)-int(n/2)):(len(sz))])
    #x = 
    s = s.reshape([nb,1])
    
    x = np.concatenate((sleft,s,sright),axis = 0)
    
    t = np.linspace(-0.5,(0.5-1/n),n)
    t = t.reshape([len(t),1])
    g =  1/sigma*np.exp(-np.pi/sigma**2*t**2)
    
    #% Initialization
    STFT = np.zeros([neta,nb],dtype=complex)
    SST2 = np.zeros([neta,nb],dtype=complex)
    omega = np.zeros([neta,nb])
    tau2 = np.zeros([neta,nb],dtype=complex)
    omega2 = np.zeros([neta,nb])
    phi22p = np.zeros([neta,nb],dtype=complex)
    vg = np.zeros([neta,8],dtype=complex)
    vgp = np.zeros([neta,5])
    Y = np.zeros([neta,7,7],dtype=complex)
    
    # Computes STFT and reassignment operators
    for b in range(nb):
        for i in range(8):
            tmp = (np.fft.fft(x[b:(b+n)]*((t**i)*g),axis=0))/n
            vg[:,i] = tmp[0:neta][:,0]
        tau2[:,b] = vg[:,1]/vg[:,0]
        for i in range(7):
            for j in range(7):
                if i >= j:
                    Y[:,i,j] = (vg[:,0]*vg[:,(i+1)]) - (vg[:,j]*vg[:,(i-j+1)])
                    
    #    W expressions
        W2 = 1/2/1j/np.pi*(vg[:,0]**2+vg[:,0]**vgp[:,1]-vg[:,1]**vgp[:,0])
    #     operator omega
        omega[:,b] = (ft-1)-np.real(vgp[:,0]/2/1j/np.pi/vg[:,0])
    
    #     operator hat p: estimations of frequency modulation  
    #    SST2
        phi22p[:,b] = W2/Y[:,1,1]
        omega2[:,b] = omega[:,b] + np.real(phi22p[:,b]*tau2[:,b])
    
    #    Storing STFT
        STFT[:,b] = vg[:,1] * np.exp(1j*np.pi*(ft-1))
            
    #%STFT=STFT*sum(g)/(n/2);
    STFT=STFT*sigma*2
    # Reassignment step
    for b in range(nb):
        for eta in range (neta):
            if abs(STFT[eta,b])>0.001*gamma:
                k = 1+round(omega2[eta,b])
                if k >=1 and k<= neta-1:
                    SST2[k,b]  = SST2[k,b] + STFT[eta,b]
    return SST2