import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import time

file_name="losses_23_11_33.csv"
csv_file_directory="../../results/training_losses/"+file_name

data = genfromtxt(csv_file_directory, delimiter=',')

reconstruction_loss=data[:,0]
soft_n_cut_loss=data[:,1]


counter=np.arange(0, len(reconstruction_loss),step=1)


plt.plot(counter,reconstruction_loss)
plt.xlabel("Iterations Number")
plt.ylabel("Reconstruction Loss")
plt.title("Reconstruction Loss vs Iterations")
plt.show()

plt.figure()
plt.plot(counter,soft_n_cut_loss)
plt.xlabel("Iterations Number")
plt.ylabel("Soft N-Cut Loss")
plt.title("Soft N-Cut Loss vs Iterations")
plt.show()