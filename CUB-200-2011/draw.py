import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(6, 4.5))

train_loss_1 = np.load('data/result/expirement1/lr=0.005/train_loss.npy')
train_loss_2 = np.load('data/result/expirement1/batchsize=16/train_loss.npy')
train_loss_3 = np.load('data/result/expirement1/batchsize=32/train_loss.npy')
x = range(1, len(train_loss_1) + 1)

plt.plot(x, train_loss_1, label='batch_size=8')
plt.plot(x, train_loss_2, label='batch_size=16')
plt.plot(x, train_loss_3, label='batch_size=32')


# plt.xticks(x)
plt.grid()
plt.ylabel('loss', fontsize=12)
plt.xlabel('epochs', fontsize=12)
plt.legend(fontsize=12)
plt.show()
