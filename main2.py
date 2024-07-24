import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from MyNet import MyCNNNet,MyNNNet
from train_test2 import train_model, test_model
import numpy as np
from torch.utils.data import Subset

#Set hyper-parameters
Batch_size=4

DEVICE = torch.device("cuda")
total_epochs=1000

#Set data transforms
pipeline = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = (0.1307,),std = (0.3081,))])

#Download MNIST dataset
train_data=datasets.CIFAR10("data",train=True,download=True,transform=pipeline)
test_data=datasets.CIFAR10("data",train=True,download=True,transform=pipeline)


subset_indices = list(range(20000))
subset_train_data = Subset(train_data, subset_indices)
subset_test_data = Subset(test_data, subset_indices)


#Create dataloader
train_loader=DataLoader(subset_train_data,batch_size=Batch_size,shuffle=True)
test_loader=DataLoader(subset_test_data,batch_size=Batch_size,shuffle=True)

#Set optimizer
model=MyCNNNet().to(DEVICE)
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
piture_index2=0
seve_epoch =100
seve_epoch2 =10
TP_cls_r = []
FP_cls_r = []
TN_cls_r = []
FN_cls_r = []
TP_cls_e = []
FP_cls_e = []
TN_cls_e = []
FN_cls_e = []
# Start train
for epoch in range(1,total_epochs+1):
    train_a, train_p, train_r, train_loss,TP_r,FP_r,TN_r,FN_r = train_model(model, DEVICE, train_loader, optimizer)
    for i in range(1,Batch_size+1):
        print(f"EPOCH_TRAIN:[{epoch}/{total_epochs}] BATCH:[{i}/{Batch_size}] Accuracy:{train_a[i-1]:.2f} Precision:{train_p[i-1]:.2f} Recall:{train_r[i-1]:.2f}")
    test_a, test_p, test_r, test_loss,TP_e,FP_e,TN_e,FN_e = test_model(model, DEVICE, test_loader)
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
    TP_cls_r.append(TP_r)
    FP_cls_r.append(FP_r)
    TN_cls_r.append(TN_r)
    FN_cls_r.append(FN_r)
    TP_cls_e.append(TP_e)
    FP_cls_e.append(FP_e)
    TN_cls_e.append(TN_e)
    FN_cls_e.append(FN_e)
    if epoch % seve_epoch == 0:

        plt.figure()

        plt.plot(EPOCH[piture_index * seve_epoch:piture_index * seve_epoch + seve_epoch],
                 train_losses[piture_index * seve_epoch:piture_index * seve_epoch + seve_epoch],label="Train_loss")
        plt.plot(EPOCH[piture_index * seve_epoch:piture_index * seve_epoch + seve_epoch],
                 test_losses[piture_index * seve_epoch:piture_index * seve_epoch + seve_epoch],label="Test_loss")
        plt.legend()


        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f"result2\\piture1\\Loss_{piture_index + 1}.png")
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
        plt.savefig(f"result2\\piture1\\Precision and Recall_{piture_index + 1}.png")
        # plt.show()
        plt.close()
        piture_index += 1

#####################################
    if epoch % seve_epoch2 == 0:


        TP_chose = TP_cls_r[piture_index2 * 10:piture_index2 * 10 + 10]
        FP_chose = FP_cls_r[piture_index2 * 10:piture_index2 * 10 + 10]
        TN_chose = TN_cls_r[piture_index2 * 10:piture_index2 * 10 + 10]
        FN_chose = FN_cls_r[piture_index2 * 10:piture_index2 * 10 + 10]

        B_FPR = []
        B_TPR = []

        for cls in range(10):
            FPR = []
            TPR = []
            for batch in range(10):
                if (FP_chose[batch][cls] + TN_chose[batch][cls])==0:
                    fpr=0
                else:
                    fpr = FP_chose[batch][cls] / (FP_chose[batch][cls] + TN_chose[batch][cls])
                if (TP_chose[batch][cls] + FN_chose[batch][cls])==0:
                    tpr=0
                else:
                    tpr = FP_chose[batch][cls] / (TP_chose[batch][cls] + FN_chose[batch][cls])
                FPR.append(fpr)
                TPR.append(tpr)
            B_FPR.append(FPR)
            B_TPR.append(TPR)
        sorted_pairs = sorted(zip(B_FPR, B_TPR))
        B_FPR, B_TPR = zip(*sorted_pairs)
        roc_data = []


        for i in range(len(B_FPR)):
            roc_data.append([B_FPR[i], B_TPR[i]])
##############################################################
        num_plots = 10  # 指定要绘制的子图数量

        # 创建一个大图
        fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))

        # 确保axs是一个一维数组，便于遍历
        if axs.ndim == 2:
            axs = axs.flatten()

        # 绘制每个子图
        for i in range(num_plots):
            fpr = roc_data[i][0]
            tpr = roc_data[i][1]

            # 绘制ROC曲线
            axs[i].plot(fpr, tpr)

            axs[i].set_xlabel('False Positive Rate')
            axs[i].set_ylabel('True Positive Rate')
            axs[i].grid(True)
            axs[i].set_title(f'Subplot {i + 1}')
            axs[i].set_xlabel('False Positive Rate')
            axs[i].set_ylabel('True Positive Rate')
            axs[i].set_title(f"class{i}")

        # 调整子图之间的间距
        plt.tight_layout()
        plt.savefig(f"result2\\piture2\\Train_ROC{piture_index2 + 1}.png")
        # 显示图形
        # plt.show()


        ####################
        TP_chose = TP_cls_e[piture_index2 * 10:piture_index2 * 10 + 10]
        FP_chose = FP_cls_e[piture_index2 * 10:piture_index2 * 10 + 10]
        TN_chose = TN_cls_e[piture_index2 * 10:piture_index2 * 10 + 10]
        FN_chose = FN_cls_e[piture_index2 * 10:piture_index2 * 10 + 10]

        B_FPR = []
        B_TPR = []

        for cls in range(10):
            FPR = []
            TPR = []
            for batch in range(10):
                if (FP_chose[batch][cls] + TN_chose[batch][cls]) == 0:
                    fpr = 0
                else:
                    fpr = FP_chose[batch][cls] / (FP_chose[batch][cls] + TN_chose[batch][cls])
                if (TP_chose[batch][cls] + FN_chose[batch][cls]) == 0:
                    tpr = 0
                else:
                    tpr = FP_chose[batch][cls] / (TP_chose[batch][cls] + FN_chose[batch][cls])
                FPR.append(fpr)
                TPR.append(tpr)
            B_FPR.append(FPR)
            B_TPR.append(TPR)
        sorted_pairs = sorted(zip(B_FPR, B_TPR))
        B_FPR, B_TPR = zip(*sorted_pairs)
        roc_data = []

        for i in range(len(B_FPR)):
            roc_data.append([B_FPR[i], B_TPR[i]])

        # 创建一个大图
        fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))

        # 确保axs是一个一维数组，便于遍历
        if axs.ndim == 2:
            axs = axs.flatten()
        num_plots = 10
        # 绘制每个子图
        for i in range(num_plots):
            fpr = roc_data[i][0]
            tpr = roc_data[i][1]

            # 绘制ROC曲线
            axs[i].plot(fpr, tpr)

            axs[i].set_xlabel('False Positive Rate')
            axs[i].set_ylabel('True Positive Rate')
            axs[i].grid(True)
            axs[i].set_title(f'Subplot {i + 1}')
            axs[i].set_xlabel('False Positive Rate')
            axs[i].set_ylabel('True Positive Rate')
            axs[i].set_title(f"class{i}")
        # plt.show()
        plt.savefig(f"result2\\piture2\\Test_ROC{piture_index2 + 1}.png")
        piture_index2+=1

#Save model
torch.save(model.state_dict(),'model.ckpt')
