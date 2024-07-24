import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from MyNet import MyNNNet
from train_test1 import train_model, test_model
import numpy as np
from torch.utils.data import Subset

#Set hyper-parameters
Batch_size=4

DEVICE = torch.device("cuda")
total_epochs=1000

#Set data transforms
pipeline = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = (0.1307,),std = (0.3081,))])

#Download MNIST dataset
train_data=datasets.MNIST("data",train=True,download=True,transform=pipeline)
test_data=datasets.MNIST("data",train=True,download=True,transform=pipeline)


subset_indices = list(range(2000))
subset_train_data = Subset(train_data, subset_indices)
subset_test_data = Subset(test_data, subset_indices)


#Create dataloader
train_loader=DataLoader(subset_train_data,batch_size=Batch_size,shuffle=True)
test_loader=DataLoader(subset_test_data,batch_size=Batch_size,shuffle=True)

#Set optimizer
model=MyNNNet().to(DEVICE)
optimizer=optim.Adam(model.parameters())

train_losses = []
test_losses = []
train_Accuracy = []
test_Accuracy = []
train_Precision = []
test_Precision = []
train_Recall = []
test_Recall = []
EPOCH = [i for i in range(1,1001)]
piture_index=0
seve_epoch =100
# Start train
for epoch in range(1,total_epochs+1):
    train_a, train_p, train_r, train_loss = train_model(model, DEVICE, train_loader, optimizer)
    for i in range(1,Batch_size+1):
        print(f"EPOCH_TRAIN:[{epoch}/{total_epochs}] BATCH:[{i}/{Batch_size}] Accuracy:{train_a[i-1]:.2f} Precision:{train_p[i-1]:.2f} Recall:{train_r[i-1]:.2f}")
    test_a, test_p, test_r, test_loss = test_model(model, DEVICE, test_loader)
    for i in range(1,Batch_size+1):
        print(f"EPOCH_TEST :[{epoch}/{total_epochs}] BATCH:[{i}/{Batch_size}] Accuracy:{test_a[i-1]:.2f} Precision:{test_p[i-1]:.2f} Recall:{test_r[i-1]:.2f}")


    train_losses.append((sum(train_loss)/len(train_loss)).cpu().item())
    test_losses.append((sum(test_loss) /len(test_loss)).cpu().item())
    train_Accuracy.append(sum(train_a)/len(train_a))
    test_Accuracy.append(sum(test_a) / len(test_a))
    train_Precision.append(sum(train_p) / len(train_p))
    test_Precision.append(sum(test_p) /len(test_p))
    train_Recall.append(sum(train_r)/len((train_r)))
    test_Recall.append(sum(test_r) / len(test_r))
    if epoch % seve_epoch == 0:

        plt.figure()

        plt.plot(EPOCH[piture_index * seve_epoch:piture_index * seve_epoch + seve_epoch],
                 train_losses[piture_index * seve_epoch:piture_index * seve_epoch + seve_epoch],label="Train_loss")
        plt.plot(EPOCH[piture_index * seve_epoch:piture_index * seve_epoch + seve_epoch],
                 test_losses[piture_index * seve_epoch:piture_index * seve_epoch + seve_epoch],label="Test_loss")
        plt.legend()


        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f"result\\Loss_{piture_index + 1}.png")
        # plt.show()
        plt.close()

        ytick_positions = np.arange(0, 1.0, 10)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        ax1.plot(EPOCH[piture_index*seve_epoch:piture_index*seve_epoch+seve_epoch],
                 train_Precision[piture_index*seve_epoch:piture_index*seve_epoch+ seve_epoch],label="Train_Precision")
        ax1.plot(EPOCH[piture_index * seve_epoch:piture_index * seve_epoch + seve_epoch],
                 train_Recall[piture_index * seve_epoch:piture_index * seve_epoch+ seve_epoch], label="Train_Recall")

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('')
        ax1.legend()

        ax2.plot(EPOCH[piture_index * seve_epoch:piture_index * seve_epoch + seve_epoch],
                 test_Precision[piture_index * seve_epoch:piture_index * seve_epoch+ seve_epoch], label="Test_Precision")
        ax2.plot(EPOCH[piture_index * seve_epoch:piture_index * seve_epoch + seve_epoch],
                 test_Recall[piture_index * seve_epoch:piture_index * seve_epoch+ seve_epoch], label="Test_Recall")

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('')
        ax2.legend()
        plt.savefig(f"result\\Precision and Recall_{piture_index + 1}.png")
        # plt.show()
        plt.close()
        piture_index += 1

#Save model
torch.save(model.state_dict(),'model.ckpt')
