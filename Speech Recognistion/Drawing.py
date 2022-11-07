import pickle
import numpy as np
a_file = open("train.pickle", "rb")
train_loss_vector, test_loss_vector, wer_calc, cer_calc = pickle.load(a_file)
a_file.close()

import matplotlib.pyplot as plt
import matplotlib


fig = plt.figure()
ax = fig.gca()
ax.plot(train_loss_vector, linewidth=5)
plt.title('Train Loss')
#matplotlib.rc('Train Loss', size=14)

fig = plt.figure()
ax = fig.gca()
ax.plot(test_loss_vector, linewidth=5)
plt.title('Test Loss')
#matplotlib.rc('Test Loss', size=14)

fig = plt.figure()
ax = fig.gca()

ax.plot(1-np.array(cer_calc), linewidth=5)
plt.title('CER (%)')
#matplotlib.rc('Test Accuracy', size=14)

