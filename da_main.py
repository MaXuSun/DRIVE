import argparse
from scipy.spatial.distance import cdist
from operator import itemgetter
import os,random,torch
import os.path as osp
import numpy as np
import torch.nn as nn
import torch.optim as optim
import models.da_network as network
from loss import Lim
from factory import Factory
from utils.run_util import lr_scheduler,op_copy,object_cal_acc,to_cuda
from utils.util import print_args
from tensorboardX import SummaryWriter
import copy

def train_source(args):
    if args.net[0:3] == 'res':
        netF = network.ResEncoder(res_name=args.net,with_bootle=False).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGEncoder(vgg_name=args.net,with_bootle=False).cuda()

    netB = network.FeatBootleneck(feature_dim=netF.in_features,type=args.classifier,
                                   bottleneck_dim=args.bottleneck_dim,use_tanh=False).cuda()
    netC = network.Classifier(args.config["class_num"], bottleneck_dim=args.bottleneck_dim,type=args.layer).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate * 0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer, = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(args.loaders["s_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(args.loaders["s_tr"])
            inputs_source, labels_source = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        outputs_source = netC(netB(netF(inputs_source)))
        classifier_loss = nn.CrossEntropyLoss()(outputs_source,labels_source)
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            acc_s_te = object_cal_acc(args.loaders["s_te"], netF, netB, netC, flag=False)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = copy.deepcopy(netF.state_dict())
                best_netB = copy.deepcopy(netB.state_dict())
                best_netC = copy.deepcopy(netC.state_dict())

            netF.train()
            netB.train()
            netC.train()

    torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    torch.save(best_netB, osp.join(args.output_dir_src, "source_B.pt"))
    torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return netF,netB,netC

def train_target(args):
    if args.net[0:3] == 'res':
        netF = network.ResEncoder(res_name=args.net,with_bootle=False).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGEncoder(vgg_name=args.net,with_bootle=False).cuda()

    netB = network.FeatBootleneck(feature_dim=netF.in_features,type=args.classifier,
                                   bottleneck_dim=args.bottleneck_dim,use_tanh=False).cuda()
    netC = network.Classifier(args.config["class_num"], bottleneck_dim=args.bottleneck_dim,type=args.layer).cuda()
    netIV = network.FeatBootleneck(feature_dim=netF.in_features,type=args.classifier,
                                   bottleneck_dim=args.bottleneck_dim,use_tanh=False).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))

    param_group = []
    param_group_FBC = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 0.1}]
        param_group_FBC += [{'params': v, 'lr': args.lr * 0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 1.0}]
        param_group_FBC += [{'params': v, 'lr': args.lr}]

    param_group_IV = []
    for k,v in netIV.named_parameters():
        param_group_IV += [{"params":v, "lr": args.lr}]
    param_group_C = []
    for k, v in netC.named_parameters():
        param_group_C += [{'params': v, 'lr': args.lr}]
        param_group_FBC += [{'params': v, 'lr': args.lr}]

    optimizer_FB = optim.SGD(param_group)
    optimizer_IV = optim.SGD(param_group_IV)
    optimizer_C = optim.SGD(param_group_C)
    optimizer_FBC = optim.SGD(param_group_FBC)
    optimizer_FB,optimizer_IV,optimizer_C,optimizer_FBC = op_copy(optimizer_FB,optimizer_IV,optimizer_C,optimizer_FBC)

    lossf_iv = nn.MSELoss().cuda()
    lossf_ce = nn.CrossEntropyLoss()
    lossf_im = Lim(args.epsion)

    max_iter = args.max_epoch * len(args.loaders["t_tr"])
    interval_iter = max_iter // args.iter_step
    iter_num = 0

    while iter_num < max_iter:
        try:
            inputs, _, pths = iter_test.next()
        except:
            iter_test = iter(args.loaders["t_tr"])
            inputs, _, pths = iter_test.next()

        if inputs.size(0) == 1:
            continue

        if iter_num % interval_iter == 0:
            netF.eval()
            netB.eval()
            netC.eval()
            mem_label = obtain_label(args.loaders["t_te"], netF, netB, netC, args)
            args.loaders["align"].construct(mem_label)
            netF.train()
            netB.train()
            netC.train()

        inputs = inputs.cuda()

        iter_num += 1
        _,now_lr = lr_scheduler(optimizer_FB, iter_num=iter_num, max_iter=max_iter)

        """train netF and netB by using target ce"""
        features_test = netB(netF(inputs))
        outputs_test = netC(features_test)

        if args.cls_par > 0:
            pred = torch.tensor(itemgetter(*pths)(mem_label), dtype=torch.long).cuda()
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= args.cls_par
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        im_loss = lossf_im(outputs_test)
        classifier_loss += im_loss
        classifier_loss *= args.w1

        optimizer_FB.zero_grad()
        classifier_loss.backward()

        try:
            s_imgs,s_labels,t_imgs,t_pths,label_t = iter_align.next()
        except:
            iter_align = iter(args.loaders["align"])
            s_imgs,s_labels,t_imgs,t_pths,label_t = iter_align.next()
        lr_scheduler(optimizer_FB,iter_num,max_iter)
        s_imgs, t_imgs, s_labels, label_t = to_cuda(s_imgs, t_imgs, s_labels, label_t)

        if args.w3 > 0:
            """train netIV"""
            s_out_fea, t_out_fea = netB(netF(s_imgs)), netIV(netF(t_imgs))
            loss_iv = lossf_iv(s_out_fea, t_out_fea) * args.w3
            optimizer_IV.zero_grad()
            loss_iv.backward()
            optimizer_IV.step()
            args.writer.add_scalar("stage2/train_IV/loss_iv", loss_iv, iter_num - 1)
            """train net C"""
            lr_scheduler(optimizer_C,iter_num,max_iter)
            ivt_out = netC(netIV(netF(t_imgs)))
            loss_ce_ivt = 0.5 * lossf_ce(ivt_out, s_labels)
            optimizer_C.zero_grad()
            loss_ce_ivt.backward()
            optimizer_C.step()
            args.writer.add_scalar("stage2/train_C/loss_ce_ivt", loss_ce_ivt, iter_num - 1)

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if args.d_name != "visdac":
                accs = {}
                for i in range(len(args.config["domains"])):
                    if i != args.s_da_i:
                        name = args.config["domains"][args.s_da_i][0].upper()+args.config["domains"][i][0].upper()
                        accs[name] = "{:.2f}".format(object_cal_acc(args.loaders[i],netF,netB,netC,flag=False))
                log_str = "Iter:{}/{}; Accuracy={}".format(iter_num,max_iter,accs)
            else:
                acc_test,all_acc = object_cal_acc(args.loaders["test"], netF, netB, netC, flag=True)
                log_str = 'Iter:{}/{}; Accuracy = {:.2f}%//{}'.format(iter_num, max_iter, acc_test,all_acc)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            netF.train()
            netB.train()
            netC.train()

    torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
    torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
    torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))

    return netF, netB, netC

def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            pths = np.asarray(data[2],dtype=str)
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_pth = pths
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_pth = np.concatenate((all_pth,pths),0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])

    dd = cdist(all_fea, initc, "cosine")
    pred_label = dd.argmin(axis=1)

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, "cosine")
        pred_label = dd.argmin(axis=1)

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    pred_label = list(pred_label.astype("int"))
    all_pth = list(all_pth)

    return dict(zip(all_pth,pred_label))

def run_source(args):
    args.name_src = args.config["domains"][args.s_da_i][0].upper()
    if not osp.exists(args.output_dir_src):
        os.makedirs(args.output_dir_src)

    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()

    args.t_da_i = 1
    args.loaders = factory.LoaderFactory(args)
    netF,netB,netC = train_source(args)

    args.out_file = open(osp.join(args.output_dir_src, 'log_test.txt'), 'w')
    netF.eval(),netB.eval(),netC.eval()
    for i in range(len(args.config["domains"])):
        if i == args.s_da_i:
            continue
        args.t_da_i = i
        args.loaders = factory.LoaderFactory(args)
        args.name = args.config["domains"][args.s_da_i][0].upper() + args.config["domains"][args.t_da_i][0].upper()
        if args.d_name != "visdac":
            acc = object_cal_acc(args.loaders["test"], netF, netB, netC, flag=False)
            log_str = '\nTraining: Task: {}, Accuracy = {:.2f}%'.format(args.name, acc)
        else:
            mean_acc,all_acc = object_cal_acc(args.loaders["test"],netF,netB,netC,flag=True)
            log_str = '\nTraining: Task: {}, Accuracy = {:.2f}%//'.format(args.name,mean_acc,all_acc)
        args.out_file.write(log_str)
        args.out_file.flush()
        print(log_str)

def run_target(args):
    for i in range(len(args.config["domains"])):
        if i == args.s_da_i:
            continue
        args.t_da_i = i
        args.loaders = factory.LoaderFactory(args)
        add_loaders(args)

        args.output_dir_src = osp.join(args.output, args.d_name, args.config["domains"][args.s_da_i][0].upper())
        args.name = args.config["domains"][args.s_da_i][0].upper() + args.config["domains"][args.t_da_i][0].upper()
        args.output_dir = osp.join(args.output, args.d_name, args.name)
        args.writer = SummaryWriter(args.output_dir+"/runs")
        print("Task: {}".format(args.name))

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()

        train_target(args)

def add_loaders(args):
    from dataset.datasets import OneDataset
    from torch.utils.data import DataLoader
    for i in range(len(args.config["domains"])):
        if i != args.s_da_i:
            t_list = np.loadtxt(osp.join(args.config["data_list_dir"], args.config["domains"][i] + ".txt"), dtype=str)
            dataset = OneDataset(t_list,args.config["transforms"][1])
            args.loaders[i] = DataLoader(dataset,batch_size=args.bs*3, shuffle=False, num_workers=args.workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IVs-DA')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument("--s_da_i", type=int, default=0, help="the index of source domain in dataset config")
    parser.add_argument("--t_da_i", type=int, default=2, help="the index of target domain in dataset config")
    parser.add_argument('--bs', type=int, default=64, help="batch_size")
    parser.add_argument('--workers', type=int, default=4, help="number of workers")
    parser.add_argument('--d_name', type=str, default='office-home')
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck_dim', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument("--s_bs", type=int, default=4, help="align source batch_size")
    parser.add_argument("--t_bs", type=int, default=4, help="align target batch_size")
    parser.add_argument("--num_selected_classes", type=int, default=8,
                        help="the kind of classes in choosing every step")

    parser.add_argument('--iter_step', type=int, default=20, help="test acc times")
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--epsion', type=float, default=1e-5)
    parser.add_argument("--w1",default=0.1,type=float)
    parser.add_argument("--w3",default=0.0,type=float)
    args = parser.parse_args()

    factory = Factory()
    args.factory = factory
    args.config = factory.ConfigFactory(args.d_name)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    args.output_dir_src = osp.join(args.output, args.d_name, args.config["domains"][args.s_da_i][0].upper())
    if not osp.exists(osp.join(args.output_dir_src, "source_F.pt")):
        run_source(args)
    run_target(args)
