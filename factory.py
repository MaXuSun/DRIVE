from torch.utils.data import DataLoader
from dataset.datasets import *
from dataset.config import *
from dataset.loaders import ClassAlignLoader

class Factory(object):
    def ConfigFactory(self,dataset_name):
        dataset_name = dataset_name.lower()
        if dataset_name in ["office-31","office","office_31","office31"]:
            return office_31
        elif dataset_name in ["office-home","office_home","officehome"]:
            return office_home
        elif dataset_name in ["pacs"]:
            return pacs
        elif dataset_name in ["dac"]:
            return dac
        elif dataset_name in ["face_gender"]:
            return face_gender
        elif dataset_name in ["all"]:
            return pacs,office_31,office_home,dac,face_gender
        return None

    def DatasetFactory(self,args):
        config = args.config
        config["s_domain"] = config["domains"][args.s_da_i]
        config["t_domain"] = config["domains"][args.t_da_i]
        config["s_bs"] = args.s_bs
        config["t_bs"] = args.t_bs

        s_tr,s_te,t_tr,t_te = split_pth2labels(config)
        s_tr_tf,s_te_tf,t_tr_tf,t_te_tf = choose_transf(config)
        one_datasets = [
            OneDataset(s_tr,s_tr_tf),
            OneDataset(s_te,s_te_tf),
            OneDataset(t_tr,t_tr_tf,True),
            OneDataset(t_te,t_te_tf,True)
        ]

        align_dataset = AlignDataset([np.vstack((s_tr,s_te)),t_tr],[s_tr_tf,t_tr_tf],config)
        return one_datasets,align_dataset

    """Domain adaptation Loader Factory"""
    def LoaderFactory(self,args):
        one_datasets, align_dataset = self.DatasetFactory(args)
        workers = args.workers
        loaders = {}
        loaders["s_tr"] = DataLoader(one_datasets[0], args.bs, shuffle=True, num_workers=workers)      # source train
        loaders["s_te"] = DataLoader(one_datasets[1], args.bs, shuffle=False, num_workers=workers)     # source test
        loaders["t_tr"] = DataLoader(one_datasets[2], args.bs, shuffle=True, num_workers=workers)      # target train
        loaders["t_te"] = DataLoader(one_datasets[2], args.bs*3, shuffle=False, num_workers=workers)     # 用于聚类

        if args.d_name == "digit":
            loaders["test"] = DataLoader(one_datasets[3], args.bs*3, shuffle=False, num_workers=workers)  # target test
        else:
            loaders["test"] = DataLoader(one_datasets[3], args.bs*3, shuffle=False, num_workers=workers)  # target test

        loaders["align"] = ClassAlignLoader(align_dataset,args.num_selected_classes,workers,collate_fn)
        return loaders

    def DgDatasetFactory(self,args):
        target = args.config["domains"][args.t_da_i]
        datasets,val_datasets,class_datasets = [],[],[]
        for domain in args.config["domains"]:
            if domain == target: continue
            print(domain)
            pth2label_train = np.loadtxt(osp.join(args.config["data_list_dir"],domain+"_train.txt"),dtype=str)
            train_dataset = OneDataset(pth2label_train,args.config["transforms"][0])
            if args.limit_source:
                train_dataset = Subset(train_dataset,args.limit_source)
            datasets.append(train_dataset)
            class_datasets.append(ClassDataset(pth2label_train,args.config["transforms"][0],args.config,args.align_bs))
            if args.d_name == "pacs":
                pth2label_val = np.loadtxt(osp.join(args.config["data_list_dir"],domain+"_crossval.txt"),dtype=str)
                val_datasets.append(OneDataset(pth2label_val, args.config["transforms"][1]))

        train_dataset = ConcatDataset(datasets)
        align_dataset = DgAlighDataset(*class_datasets,choose_num=args.choose_num)
        pth2label_target = np.loadtxt(osp.join(args.config["data_list_dir"],target+"_test.txt"),dtype=str)
        target_dataset = OneDataset(pth2label_target,args.config["transforms"][1])
        target_dataset = Subset(target_dataset,args.limit_target)
        target_dataset = ConcatDataset([target_dataset])

        if args.d_name == "pacs":
            val_dataset = ConcatDataset(val_datasets)
        else:
            val_dataset = target_dataset
        return train_dataset,target_dataset,align_dataset,val_dataset
    def DgLoaderFactory(self,args):
        datasets = self.DgDatasetFactory(args)
        loaders = {}
        loaders["tr_loader"] = DataLoader(datasets[0],batch_size=args.bs,shuffle=True,num_workers=args.workers,pin_memory=True,drop_last=True)
        loaders["target_loader"] = DataLoader(datasets[1],batch_size=args.bs,shuffle=False,num_workers=args.workers,pin_memory=True,drop_last=False)
        loaders["align_loader"] = DataLoader(datasets[2],batch_size=args.num_selected_classes,shuffle=True,num_workers=args.workers,pin_memory=True,drop_last=False)
        loaders["val_loader"] = DataLoader(datasets[3], batch_size=args.bs, shuffle=False, num_workers=args.workers,
                                           pin_memory=True, drop_last=True)
        return loaders



def split_train_test(all_list,num_classes,ratio):
    """split train and test data"""
    train = []
    test = []
    for c in range(num_classes):
        index = np.where(all_list[:,1]==str(c))
        random.shuffle(index[0])
        split = int(len(index[0])*ratio)
        train.append(all_list[index[0][:split]])
        test.append(all_list[index[0][split:]])
    return np.vstack(train),np.vstack(test)

def split_pth2labels(config):
    """split data list file to train and test data"""

    list_dir = config["data_list_dir"]
    s_domain,t_domain = config["s_domain"],config["t_domain"]

    if config["dataset_name"] == "digit":  # 对于digit数据集，较为特殊，已经分好train和test
        s_tr = np.loadtxt(osp.join(list_dir,s_domain+"_train.txt"),dtype=str)
        s_te = np.loadtxt(osp.join(list_dir,s_domain+"_test.txt"),dtype=str)
        t_tr = np.loadtxt(osp.join(list_dir,t_domain+"_train.txt"),dtype=str)
        t_te = np.loadtxt(osp.join(list_dir,t_domain+"_test.txt"),dtype=str)
    else:                                   # 对于其他数据集，根据设置的tr_te_scale来确定source和target的train和test
        s_list = np.loadtxt(osp.join(list_dir,s_domain+".txt"),dtype=str)
        t_list = np.loadtxt(osp.join(list_dir,t_domain+".txt"),dtype=str)

        s_tr,s_te = split_train_test(s_list,config["class_num"],config["tr_te_scale"][0])
        t_tr,t_te = t_list,t_list
    return s_tr,s_te,t_tr,t_te

def choose_transf(config):
    """get the train and test transforms"""
    cts = config["transforms"]
    s_tr_tf, s_te_tf, t_tr_tf, t_te_tf = None,None,None,None
    if config["dataset_name"] == "digit":
        s_domain,t_domain = config["s_domain"],config["t_domain"]
        if s_domain == "svhn" and t_domain == "mnist":
            s_tr_tf,s_te_tf = cts[2],cts[2]
            t_tr_tf,t_te_tf = cts[3],cts[3]
        elif s_domain == "usps" and t_domain == "mnist":
            s_tr_tf,s_te_tf = cts[1],cts[1]
            t_tr_tf,t_te_tf = cts[0],cts[0]
        elif s_domain == "mnist" and t_domain == "usps":
            s_tr_tf,s_te_tf = cts[0],cts[0]
            t_tr_tf,t_te_tf = cts[0],cts[0]
    else:
        s_tr_tf,s_te_tf = cts[0],cts[1]
        t_tr_tf,t_te_tf = cts[0],cts[1]
    return s_tr_tf,s_te_tf,t_tr_tf,t_te_tf

def collate_fn(data):
    length = len(data)
    data_collate = [None]*5
    for i in range(5):
        if i != 3:
            data_collate[i] = torch.cat([data[j][i] for j in range(length)],0)
        else:
            data_collate[3] = np.concatenate([data[j][3] for j in range(length)],0)

    return data_collate



