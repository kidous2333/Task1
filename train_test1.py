import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,precision_score ,recall_score

def calculate_metrics(outputs,labels):

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for pred, act in zip(outputs, labels):
        if pred == 1 and act == 1:
            tp += 1
        elif pred == 1 and act == 0:
            fp += 1
        elif pred == 0 and act == 0:
            tn += 1
        else:
            fn += 1
    acc = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp) if (tp+fp)>0 else 0
    recall = tp/(tp+fn) if (tp+fn)>0 else 0

    return acc,precision,recall

def train_model(model,device,train_loader,optimizer):
    model.train()
    accuracy = []
    precision = []
    recall = []
    loss_all = []


    for batch_index,(data,target) in enumerate(train_loader):
        data_p = []
        data_true = []
        pre = []
        true = []
        A = []
        P = []
        R = []
        data,target=data.to(device),target.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=F.cross_entropy(output,target)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(output.data, 1)
        data_p.extend(predicted.cpu().numpy())
        data_true.extend(target.cpu().numpy())
        for i in range(10):
            pre.append([1 if j == i else 0 for j in data_p ])
            true.append([1 if j == i else 0 for j in data_true])
            acc,prec,reca = calculate_metrics(pre[i],true[i])
            A.append(acc)
            P.append(prec)
            R.append(reca)
        accuracy.append(sum(A)/len(A))
        precision.append(sum(P)/len(P))
        recall.append(sum(R)/len((R)))
        loss_all.append(loss)

    return accuracy, precision, recall, loss_all
def test_model(model,device,test_loader):
    accuracy = []
    precision = []
    recall = []
    loss_all = []
    with torch.no_grad():
        for data,target in test_loader:
            data_p = []
            data_true = []
            pre = []
            true = []
            A = []
            P = []
            R = []
            data,target=data.to(device),target.to(device)
            output=model(data)
            loss=F.cross_entropy(output,target)
            _, predicted = torch.max(output.data, 1)
            data_p.extend(predicted.cpu().numpy())
            data_true.extend(target.cpu().numpy())

            for i in range(10):
                pre.append([1 if j == i else 0 for j in data_p])
                true.append([1 if j == i else 0 for j in data_true])
                acc, prec, reca = calculate_metrics(pre[i], true[i])
                A.append(acc)
                P.append(prec)
                R.append(reca)
            accuracy.append(sum(A) / len(A))
            precision.append(sum(P) / len(P))
            recall.append(sum(R) / len((R)))
            loss_all.append(loss)

        return accuracy, precision, recall, loss_all