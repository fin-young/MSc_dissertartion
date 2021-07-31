import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

path = './log_CIFAR/best/AMSGrad_VGG11_bs{64}_lr{1.0e-03}_eps{1.0e-04}/history.pkl'
df_amsgrad = pkl.load(open(path, 'rb'))

path = './log_CIFAR/best/ADAM_VGG11_bs{64}_lr{5.0e-04}_eps{1.0e-04}/history.pkl'
df_adam = pkl.load(open(path, 'rb'))

path = './log_CIFAR/best/RMSProp_VGG11_bs{64}_lr{5.0e-04}_eps{1.0e-04}/history.pkl'
df_rms = pkl.load(open(path, 'rb'))

path = './log_CIFAR/best/THEOPOULA_VGG11_bs{64}_lr{1.0e-02}_beta{1.0e+14}_eps{1.0e-04}/history.pkl'
df_theopoula = pkl.load(open(path, 'rb'))


plt.figure(1)

for i, key in zip(range(1, 5), df_amsgrad.keys()):
    plt.figure(i)
    plt.plot(df_amsgrad[key], label='AMSGRAD')
    plt.plot(df_adam[key], label='ADAM')
    plt.plot(df_rms[key], label='RMSProp')
    plt.plot(df_theopoula[key], label='TheoPouLa')
    plt.legend()
    plt.title(key)

print('AMSGrad:  test_loss -',np.array(df_amsgrad['test_loss']).min().item())
print('ADAM:  test_loss -',np.array(df_adam['test_loss']).min().item())
print('RMSprop:  test_loss -',np.array(df_rms['test_loss']).min().item())
print('THEOPOULA:  test_loss -',np.array(df_theopoula['test_loss']).min().item())


plt.show()