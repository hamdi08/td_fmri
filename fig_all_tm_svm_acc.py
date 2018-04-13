
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "STIX"

measures_dict = sio.loadmat('acc_all_adhd.mat')
measures_array = measures_dict.get('acc_all')


# Draw fig 1 --- TM1 -- CP and Tucker
x = np.arange(1, 21, 1)
y1 = measures_array[0,0,:]
y2 = measures_array[0,1,:]

fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)

ax.plot(x, y1, color = 'r', linewidth = 2, marker = 'o', markersize=10, fillstyle = 'none', markeredgewidth = 2, label='CP')
ax.plot(x, y2, color = 'b', linewidth = 2, marker = 'x', markersize=10, markeredgewidth = 2, label='Tucker')
ax.set(title='Classification accuracy with SVM and stratified 8-fold cross validation', xlabel='Number of columns in subject factor matrix', ylabel='Accuracy')
#ax.xaxis.label.set_size(20)


plt.xticks(x)
plt.ylim(0,1)
ax.legend(loc='best')
plt.rcParams.update({'font.size': 13})

# Draw fig 2 --- TM2 -- CP and Tucker
x = np.arange(1, 21, 1)
y1 = measures_array[1,0,:]
y2 = measures_array[1,1,:]

fig = plt.figure(2, figsize=(9, 6))
ax = fig.add_subplot(111)

ax.plot(x, y1, color = 'r', linewidth = 2, marker = 'o', markersize=10, fillstyle = 'none', markeredgewidth = 2, label='CP')
ax.plot(x, y2, color = 'b', linewidth = 2, marker = 'x', markersize=10, markeredgewidth = 2, label='Tucker')
ax.set(title='Classification accuracy with SVM and stratified 8-fold cross validation', xlabel='Number of columns in subject factor matrix', ylabel='Accuracy')
plt.xticks(x)
plt.ylim(0,1)
ax.legend(loc='best')
plt.rcParams.update({'font.size': 13})


# Draw fig 3 --- TM3 -- CP and Tucker
x = np.arange(1, 21, 1)
y1 = measures_array[2,0,:]
y2 = measures_array[2,1,:]

fig = plt.figure(3, figsize=(9, 6))
ax = fig.add_subplot(111)

ax.plot(x, y1, color = 'r', linewidth = 2, marker = 'o', markersize=10, fillstyle = 'none', markeredgewidth = 2, label='CP')
ax.plot(x, y2, color = 'b', linewidth = 2, marker = 'x', markersize=10, markeredgewidth = 2, label='Tucker')
ax.set(title='Classification accuracy with SVM and stratified 8-fold cross validation', xlabel='Number of columns in subject factor matrix', ylabel='Accuracy')
plt.xticks(x)
plt.ylim(0,1)
ax.legend(loc='best')
plt.rcParams.update({'font.size': 13})


# Draw fig 4 --- TM4 -- CP and Tucker
x = np.arange(1, 21, 1)
y1 = measures_array[3,0,:]
y2 = measures_array[3,1,:]

fig = plt.figure(4, figsize=(9, 6))
ax = fig.add_subplot(111)

ax.plot(x, y1, color = 'r', linewidth = 2, marker = 'o', markersize=10, fillstyle = 'none', markeredgewidth = 2, label='CP')
ax.plot(x, y2, color = 'b', linewidth = 2, marker = 'x', markersize=10, markeredgewidth = 2, label='Tucker')
ax.set(title='Classification accuracy with SVM and stratified 8-fold cross validation', xlabel='Number of columns in subject factor matrix', ylabel='Accuracy')
plt.xticks(x)
plt.ylim(0,1)
ax.legend(loc='best')
plt.rcParams.update({'font.size': 13})


# Draw fig 5 --- TM5 -- CP and Tucker
x = np.arange(1, 21, 1)
y1 = measures_array[4,0,:]
y2 = measures_array[4,1,:]

fig = plt.figure(5, figsize=(9, 6))
ax = fig.add_subplot(111)

ax.plot(x, y1, color = 'r', linewidth = 2, marker = 'o', markersize=10, fillstyle = 'none', markeredgewidth = 2, label='CP')
ax.plot(x, y2, color = 'b', linewidth = 2, marker = 'x', markersize=10, markeredgewidth = 2, label='Tucker')
ax.set(title='Classification accuracy with SVM and stratified 8-fold cross validation', xlabel='Number of columns in subject factor matrix', ylabel='Accuracy')
plt.xticks(x)
plt.ylim(0,1)
ax.legend(loc='best')
plt.rcParams.update({'font.size': 13})


