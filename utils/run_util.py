from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import numpy as np

def op_copy(*optimizers):
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr0'] = param_group['lr']
    return optimizers

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    now_lr = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        now_lr = param_group["lr"]
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer,now_lr

def to_cuda(*parms):
    if not torch.cuda.is_available():
        return parms
    results = []
    for item in parms:
        results.append(item.cuda())
    return results

def object_cal_acc(loader, netF, netB, netC,netIV=None,flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if netIV:
                outputs = netC(netIV(netB(netF(inputs))))
            else:
                outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)  # [N]

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    if flag:
        matrix = confusion_matrix(all_label,torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1)*100
        mean_acc = acc.mean()
        aa = [str(np.round(i,3)) for i in acc]
        acc = '-'.join(aa)
        return mean_acc,acc
    else:
        return accuracy * 100