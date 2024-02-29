#import cv2
import numpy as np

#Data = cv2.imread('662_9111_129_32_vh.jpg',0)
#Data = cv2.resize(Data,[256,256])/256
#x = Data[0,:]
def MSST(x,hlength,num):
#    x = np.transpose(Data[0,:])
#    hlength = 64
#    num = 1
    xrow = x.shape[0]
    hlength=hlength+1-hlength%2;
    ht = np.linspace(-0.5,0.5,hlength)
    
    # Gaussian window
    h = np.exp(-np.pi/(0.32**2)*(ht**2))
    hrow = h.shape[0]
    Lh=(hrow-1)/2
    N=xrow
    t=np.linspace(1,xrow,xrow)
    tcol = t.shape[0]
    
    tfr = np.zeros([N,tcol],dtype=complex)
    omega = np.zeros ([round(N/2),tcol-1])
    omega2 = np.zeros ([round(N/2),tcol])
    Ts = np.zeros ([round(N/2),tcol],dtype=complex)
    
    for icol in range(tcol):
        ti= t[icol]
        tau=np.arange(int(-min([round(N/2)-1,Lh,ti-1])),int(min([round(N/2)-1,Lh,xrow-ti]))+1)
        indices= ((N+tau)%N)
        rSig = [x[int(px-1)] for px in (ti+tau)]
        rsign_this = rSig*np.conj([h[int(px)] for px in (Lh+tau)])
        for pt in range(len(indices)):
            tfr[indices[pt],icol]=rsign_this[pt]
    
    tfr=np.fft.fft(tfr,axis=0)
    
    tfr=tfr[0:round(N/2),:]
    
    for i in range(round(N/2)):
        omega[i,:]=np.diff(np.unwrap(np.angle(tfr[i,:])))*(N)/2/np.pi
    
    omega = np.append(omega, omega[:,-1][:, np.newaxis], axis=1)
    
    omega = np.round(omega)
    
    (neta,nb)=tfr.shape
    
    
    omega2=omega
    
    for b in range(nb):
        for eta in range(neta):
            if abs(tfr[eta,b])>0.0001:
                k = int(omega2[eta,b])
                if k >= 1 and k<=neta:
                    Ts[k-1,b] = Ts[k-1,b] + tfr[eta,b]
    
    Ts=Ts/(xrow/2)
    return Ts


