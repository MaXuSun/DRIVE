from torch import optim
import os.path as osp
import torch,random,os
from torch import nn
import numpy as np
from models.dg_network import resnet50,resnet18,FeatBootleneck,Classifier,Alexnet
from factory import Factory
from loss import Lim
from tensorboardX import SummaryWriter
import argparse

def get_optim_and_scheduler(params, epochs, lr, nesterov=False):
    optimizer = optim.SGD(params, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    step_size = int(epochs * .8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    return optimizer, scheduler

def shape_data(*datas):
    """cat the data from every domain"""
    result = []
    for data in datas:
        size = [-1]
        size.extend(data.size()[2:])
        result.append(data.reshape(size))
    return result

def flip_data(*datas):
    """flip data"""
    result = []
    for data in datas:
        size = [-1]
        data_flip = torch.flip(data,(3,)).detach().clone()
        data = torch.stack((data,data_flip),dim=1)
        size.extend(data.size()[2:])
        result.append(data.reshape(size))
    return result

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.iter_step = 20
        self.output_dir = osp.join("DG_seed"+str(args.seed),args.net,args.config["domains"][args.t_da_i])
        if not osp.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.writer = SummaryWriter(self.output_dir+"/target_{}_method_{}/logs".format(args.t_da_i,args.use_ours))

        """load model"""
        if args.net == 'resnet18':
            self.netF = resnet18(pretrained=True, classes=args.config["class_num"]).cuda()
        elif args.net == 'resnet50':
            self.netF = resnet50(pretrained=True, classes=args.config["class_num"]).cuda()
        elif args.net == "alexnet":
            self.netF = Alexnet(pretrain=True).cuda()
        else:
            self.netF = resnet18(pretrained=True, classes=args.config["class_num"]).cuda()


        self.netB = FeatBootleneck(self.netF.in_features,type="wn").to(device)
        self.netC = Classifier(args.config["class_num"]).to(device)
        self.netIV1 = FeatBootleneck(self.netF.in_features, type="wn").to(device)
        self.netIV2 = FeatBootleneck(self.netF.in_features, type="wn").to(device)

        """init optimizer"""
        params_groups = self.get_params(self.netIV1, self.netIV2)
        self.optimizer_FBC, self.scheduler_FBC = get_optim_and_scheduler(params_groups[2], args.max_epoch, args.lr)
        self.optimizer_FB, self.scheduler_FB = get_optim_and_scheduler(params_groups[0], args.max_epoch, args.lr)
        self.optimizer_C, self.scheduler_C = get_optim_and_scheduler(params_groups[1],args.max_epoch,args.lr)
        self.optimizer_IV1,self.scheduler_IV1 = get_optim_and_scheduler(params_groups[3][0],args.max_epoch,args.lr)
        self.optimizer_IV2,self.scheduler_IV2 = get_optim_and_scheduler(params_groups[3][1],args.max_epoch,args.lr)

        """init loss function"""
        self.criterion_iv = nn.MSELoss()
        self.criterion_im = Lim(1e-5)
        self.criterion_ce = nn.CrossEntropyLoss()

    def train_epoch(self, epoch):
        self.netF.train(),self.netB.train(),self.netC.train()
        length = self.args.multi * len(self.args.loaders["align_loader"])
        for it in range(length):
            for i in range(int(15 * (1-epoch / self.args.max_epoch))):
                try:
                    (datas, labels), _ = iter_all.next()
                except:
                    iter_all = iter(self.args.loaders["tr_loader"])
                    (datas, labels), _ = iter_all.next()
                datas, labels = datas.to(self.device), labels.to(self.device)
                data_flip = torch.flip(datas, (3,)).detach().clone()
                datas = torch.cat((datas, data_flip))
                labels = torch.cat((labels, labels))
                """train netF,netB,netC by using loss ce"""
                out = self.netC(self.netB(self.netF(datas)))
                loss_ce = self.criterion_ce(out, labels)
                self.optimizer_FBC.zero_grad()
                loss_ce.backward()
                self.optimizer_FBC.step()
                self.writer.add_scalar("loss_ce", loss_ce, it)

        self.netF.eval(),self.netB.eval(),self.netC.eval()
        acc = self.test(self.args.loaders["target_loader"])
        print("test; Epoch: {}, acc: {:.2f}\n".format(epoch,acc))

    def train_epoch_IV(self, epoch):
        self.netF.train(),self.netB.train(),self.netC.train(),self.netIV1.train()
        length = self.args.multi * len(self.args.loaders["align_loader"])
        for it in range(length):
            for i in range(int(20*(1-epoch/self.args.max_epoch))):
                try:
                    (datas,labels),_ = iter_all.next()
                except:
                    iter_all = iter(self.args.loaders["tr_loader"])
                    (datas,labels),_ = iter_all.next()
                datas,labels = datas.to(self.device),labels.to(self.device)
                data_flip = torch.flip(datas,(3,)).detach().clone()
                datas = torch.cat((datas,data_flip))
                labels = torch.cat((labels,labels))
                """train netF,netB,netC by using loss ce"""
                out = self.netC(self.netB(self.netF(datas)))
                loss_ce = self.criterion_ce(out,labels)
                self.optimizer_FBC.zero_grad()
                loss_ce.backward()
                self.optimizer_FBC.step()
                self.writer.add_scalar("loss_ce",loss_ce,it)

            try:
                datas,labels = iter_align.next()
            except:
                iter_align = iter(self.args.loaders["align_loader"])
                datas,labels = iter_align.next()

            (data1,data2,data3),label1,label2,label3 = shape_data(*datas),labels[0].reshape(-1),labels[1].reshape(-1),labels[2].reshape(-1)
            data1,data2,data3 = data1.to(self.device),data2.to(self.device),data3.to(self.device)
            label1,label2,label3 = label1.to(self.device),label2.to(self.device),label3.to(self.device)

            """train net IV"""
            self.netF.eval(),self.netB.eval()
            out1 = self.netB(self.netF(data1))
            out2 = self.netIV1(self.netF(data2))
            out3 = self.netIV2(self.netF(data3))
            loss12_iv = 0.1*self.criterion_iv(out1,out2)
            loss13_iv = 0.9*self.criterion_iv(out1,out3)
            loss_iv = loss12_iv+loss13_iv
            self.optimizer_IV1.zero_grad()
            self.optimizer_IV2.zero_grad()
            loss_iv.backward()
            self.optimizer_IV1.step()
            self.optimizer_IV2.step()
            self.netF.train(),self.netB.train()
            self.writer.add_scalar("loss_iv", loss_iv, epoch * length + it)
            """train net C"""
            self.optimizer_C.zero_grad()
            out1 = self.netC(self.netIV1(self.netF(data1)))
            loss1 = self.criterion_ce(out1, label1) * 0.1
            loss1.backward(retain_graph=True)
            self.writer.add_scalar("loss_ce1", loss1, epoch * length + it)
            out1 = self.netC(self.netIV2(self.netF(data1)))
            loss1 = self.criterion_ce(out1, label1) * 0.9
            loss1.backward()
            self.writer.add_scalar("loss_ce2", loss1, epoch * length + it)
            self.optimizer_C.step()

            if it % 10 == 0 and it !=0:
                self.netF.eval(), self.netB.eval(), self.netC.eval()
                acc = self.test(self.args.loaders["target_loader"])
                print("test; Epoch: {}, acc: {:.2f}\n".format(epoch, acc))
                self.netF.train(),self.netB.train(),self.netC.train()

        self.netF.eval(), self.netB.eval(), self.netC.eval()
        acc = self.test(self.args.loaders["target_loader"])
        print("test; Epoch: {}, acc: {:.2f}\n".format(epoch, acc))

    def pre_train(self):
        best_netF,best_netB,best_netC = None,None,None
        self.netF.train(),self.netB.train(),self.netC.train()
        max_iter = 20 * len(self.args.loaders["tr_loader"])
        max_acc = 0
        for it in range(max_iter):
            try:
                (datas,labels),_ = iter_all.next()
            except:
                iter_all = iter(self.args.loaders["tr_loader"])
                (datas,labels),_ = iter_all.next()
            datas,labels = datas.to(self.device),labels.to(self.device)
            data_flip = torch.flip(datas,(3,)).detach().clone()
            datas = torch.cat((datas,data_flip))
            labels = torch.cat((labels,labels))
            """train netF,netB,netC by using loss ce"""
            out = self.netC(self.netB(self.netF(datas)))
            loss_ce = self.criterion_ce(out,labels)
            self.optimizer_FBC.zero_grad()
            loss_ce.backward()
            self.optimizer_FBC.step()
            self.writer.add_scalar("loss_ce",loss_ce,it)

            if (it % len(self.args.loaders["tr_loader"]) == (len(self.args.loaders["tr_loader"])//2)) or it == max_iter:
                self.netF.eval(),self.netB.eval(),self.netC.eval()
                acc = self.test(self.args.loaders["val_loader"])
                print("val: iter_num: {}/{}, acc: {:.2f}\n".format(it,max_iter,acc))
                if acc > max_acc:
                    max_acc = acc
                    best_netF = self.netF.state_dict()
                    best_netB = self.netB.state_dict()
                    best_netC = self.netC.state_dict()
                self.netF.train(),self.netB.train(),self.netC.eval()
        if self.args.d_name == "pacs":
            torch.save(best_netF,osp.join(self.output_dir,"source_F.pt"))
            torch.save(best_netB,osp.join(self.output_dir,"source_B.pt"))
            torch.save(best_netC,osp.join(self.output_dir,"source_C.pt"))
        else:
            torch.save(self.netF.state_dict(), osp.join(self.output_dir, "source_F.pt"))
            torch.save(self.netB.state_dict(), osp.join(self.output_dir, "source_B.pt"))
            torch.save(self.netC.state_dict(), osp.join(self.output_dir, "source_C.pt"))

        self.netF.eval(), self.netB.eval(), self.netC.eval()
        acc = self.test(self.args.loaders["target_loader"])
        print("target: acc: {:.2f}\n".format(acc))

    def test(self, loader):
        total = 0
        with torch.no_grad():
            correct = 0
            for it, ((data, labels), _) in enumerate(loader):
                data, labels = data.to(self.device), labels.to(self.device)
                class_logit = self.netC(self.netB(self.netF(data)))
                _, cls_pred = class_logit.max(dim=1)
                correct += torch.sum(cls_pred == labels.data)
                total += data.size(0)
        return (float(correct)/total)*100

    def get_params(self,*netIVs):
        params_group_FBC = []
        params_group_FB = []
        params_group_C = []
        for k, v in self.netF.named_parameters():
            params_group_FBC += [{"params": v, "lr": self.args.lr*0.1}]
            params_group_FB += [{"params": v, "lr": self.args.lr*0.1}]
        for k, v in self.netB.named_parameters():
            params_group_FBC += [{"params": v, "lr": self.args.lr}]
            params_group_FB += [{"params": v, "lr": self.args.lr}]
        for k, v in self.netC.named_parameters():
            params_group_FBC += [{"params": v, "lr": self.args.lr}]
            params_group_C += [{'params': v, 'lr': self.args.lr}]

        params_group_IVs = []
        for netIV in netIVs:
            params_group_IV = []
            for k, v in netIV.named_parameters():
                params_group_IV += [{"params": v, "lr": self.args.lr}]
            params_group_IVs.append(params_group_IV)

        return params_group_FB, params_group_C, params_group_FBC, params_group_IVs

    def train(self):
        """pre train"""
        if not osp.exists(osp.join(self.output_dir,"source_F.pt")):
            self.pre_train()

        """train"""
        self.netF.load_state_dict(torch.load(osp.join(self.output_dir, "source_F.pt")))
        self.netB.load_state_dict(torch.load(osp.join(self.output_dir, "source_B.pt")))
        self.netC.load_state_dict(torch.load(osp.join(self.output_dir, "source_C.pt")))

        for self.current_epoch in range(self.args.max_epoch):
            self.scheduler_FB.step()
            self.scheduler_C.step()
            self.scheduler_FBC.step()
            self.scheduler_IV1.step()
            self.scheduler_IV2.step()
            if self.args.use_ours == 1:
                self.train_epoch(self.current_epoch)
            elif self.args.use_ours == 2:
                self.train_epoch_IV(self.current_epoch)

def main():
    parser = argparse.ArgumentParser(description='IVs_DG')
    parser.add_argument("--d_name", type=str, default="pacs", help="the using dataset name")
    parser.add_argument("--t_da_i", type=int, default=2, help="the index of target domain in dataset config")
    parser.add_argument("--bs", type=int, default=64, help="batch_size")
    parser.add_argument("--align_bs", type=int, default=4, help="align source batch_size")
    parser.add_argument("--num_selected_classes", type=int, default=4,
                        help="the kind of classes in choosing every step")
    parser.add_argument("--use_ours", type=int, default=1)
    parser.add_argument("--choose_num", type=int, default=2)
    parser.add_argument('--max_epoch', type=int, default=20, help="maximum epoch")
    parser.add_argument('--step', type=int, default=15, help="pre step")
    parser.add_argument('--workers', type=int, default=4, help="number of workers")
    parser.add_argument('--multi', type=int, default=10, help="multi ratio")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument("--net", type=str, default="resnet18", help="resnet18 resnet50")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    factory = Factory()
    args.config = factory.ConfigFactory(args.d_name)
    args.loaders = factory.DgLoaderFactory(args)
    print("Target: {}".format(args.config["domains"][args.t_da_i]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    trainer.train()

if __name__ == "__main__":
    main()