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
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return acc,precision,recall,tp, fp, tn, fn

def train_model(model,device,train_loader,optimizer):
    model.train()
    accuracy = []
    precision = []
    recall = []
    TP = []
    FP = []
    TN = []
    FN = []
    loss_all = []
    TPs=[]
    FPs=[]
    TNs=[]
    FNs=[]

    for batch_index,(data,target) in enumerate(train_loader):
        data_p = []
        data_true = []
        pre = []
        true = []
        tps = []
        fps = []
        tns = []
        fns = []
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
            acc,prec,reca,tp, fp, tn, fn = calculate_metrics(pre[i],true[i])
            A.append(acc)
            P.append(prec)
            R.append(reca)
            tps.append(tp)
            fps.append(fp)
            tns.append(tn)
            fns.append(fn)
        TPs.append(tps)
        FPs.append(fps)
        TNs.append(tns)
        FNs.append(fns)
        accuracy.append(sum(A) / len(A))
        precision.append(sum(P) / len(P))
        recall.append(sum(R) / len((R)))
        A.append(acc)
        P.append(prec)
        R.append(reca)
    for cls in range(10):
        tp_=0
        fp_=0
        tn_=0
        fn_=0
        for i in range(4):
            tp_ += TPs[i][cls]
            fp_ += FPs[i][cls]
            tn_ += TNs[i][cls]
            fn_ += FNs[i][cls]
        TP.append(tp_)
        FP.append(fp_)
        TN.append(tn_)
        FN.append(fn_)

        loss_all.append(loss)

    return accuracy, precision, recall, loss_all,TP,FP,TN,FN
def test_model(model,device,test_loader):
    accuracy = []
    precision = []
    recall = []
    TP = []
    FP = []
    TN = []
    FN = []
    loss_all = []
    TPs = []
    FPs = []
    TNs = []
    FNs = []

    for batch_index, (data, target) in enumerate(test_loader):
        data_p = []
        data_true = []
        pre = []
        true = []
        tps = []
        fps = []
        tns = []
        fns = []
        A = []
        P = []
        R = []
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.cross_entropy(output, target)
        _, predicted = torch.max(output.data, 1)
        data_p.extend(predicted.cpu().numpy())
        data_true.extend(target.cpu().numpy())
        for i in range(10):
            pre.append([1 if j == i else 0 for j in data_p])
            true.append([1 if j == i else 0 for j in data_true])
            acc, prec, reca, tp, fp, tn, fn = calculate_metrics(pre[i], true[i])
            A.append(acc)
            P.append(prec)
            R.append(reca)
            tps.append(tp)
            fps.append(fp)
            tns.append(tn)
            fns.append(fn)
        TPs.append(tps)
        FPs.append(fps)
        TNs.append(tns)
        FNs.append(fns)
        accuracy.append(sum(A) / len(A))
        precision.append(sum(P) / len(P))
        recall.append(sum(R) / len((R)))
        A.append(acc)
        P.append(prec)
        R.append(reca)
    for cls in range(10):
        tp_ = 0
        fp_ = 0
        tn_ = 0
        fn_ = 0
        for i in range(4):
            tp_ += TPs[i][cls]
            fp_ += FPs[i][cls]
            tn_ += TNs[i][cls]
            fn_ += FNs[i][cls]
        TP.append(tp_)
        FP.append(fp_)
        TN.append(tn_)
        FN.append(fn_)

        loss_all.append(loss)

    return accuracy, precision, recall, loss_all, TP, FP, TN, FN