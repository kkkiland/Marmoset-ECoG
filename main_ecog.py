import os
import argparse
import logging
import time
import scipy.io as sio
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import Setup, Initialization, dataset_class
from Models.shapeformer import model_factory
from Models.optimizers import get_optimizer
from Models.loss import get_loss_module
from Models.utils import load_model
from Training import SupervisedTrainer, train_runner
from Shapelet.mul_shapelet_discovery import ShapeletDiscover
import torch
import random
from tqdm import tqdm
import os
import platform
import torch
from tqdm import tqdm
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

logger = logging.getLogger('__main__')
parser = argparse.ArgumentParser()
# -------------------------------------------- Input and Output --------------------------------------------------------
parser.add_argument('--data_path', default='new_data/', choices={'new_data/'}, help='Data path')
parser.add_argument('--output_dir', default='Results',
                    help='Root output directory. Must exist. Time-stamped directories will be created inside.')
parser.add_argument('--Norm', type=bool, default=False, help='Data Normalization')
parser.add_argument('--num_train', type=int, default=100, help="Number of samples for training shapelet")
parser.add_argument('--print_interval', type=int, default=10, help='Print batch info every this many batches')
# ----------------------------------------------------------------------------------------------------------------------
parser.add_argument("--dataset_pos", default=10, type=int, help="position of datasets")
# ------------------------------------- Dataset-------------------------------------------------------------------------

parser.add_argument("--num_shapelet", default=50, type=int, help="number of shapelets")
parser.add_argument("--window_size", default=20, type=int, help="window size")

# ------------------------------------- Model Parameter and Hyperparameter ---------------------------------------------
parser.add_argument('--Net_Type', default=['Shapeformer'], choices={'Shapeformer'})
# Local Information
parser.add_argument("--len_w", default=64, type=float, help="window size")
parser.add_argument("--local_embed_dim", default=48, type=int, help="embedding dimension of shape")
parser.add_argument("--local_pos_dim", default=48, type=int, help="embedding dimension of pos")

# Global Information
parser.add_argument("--num_pip", default=0.2, type=float, help="number of pips")
parser.add_argument("--sge", default=0, type=int, help="stop-gradient epochs")
parser.add_argument("--shape_embed_dim", default=128, type=int, help="embedding dimension of shape")
parser.add_argument("--pos_embed_dim", default=128, type=int, help="embedding dimension of pos")
parser.add_argument("--processes", default=1, type=int, help="number of processes for extracting shapelets")
parser.add_argument("--pre_shapelet_discovery", default=1, type=int, help="number of processes for extracting shapelets")

# Transformers Parameters
parser.add_argument('--emb_size', type=int, default=64, help='Internal dimension of transformer embeddings')
parser.add_argument('--dim_ff', type=int, default=256, help='Dimension of dense feedforward part of transformer layer')
parser.add_argument('--num_heads', type=int, default=2, help='Number of multi-headed attention heads')
parser.add_argument('--local_num_heads', type=int, default=4, help='Number of multi-headed attention heads')
parser.add_argument('--dropout', type=float, default=0.5, help='Droupout regularization ratio')

# Training Parameters/ Hyper-Parameters
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='learning rate')
parser.add_argument('--key_metric', choices={'loss', 'accuracy', 'precision', 'AUROC'}, default='AUROC',
                    help='Metric used for defining best epoch')

parser.add_argument('--gpu', type=int, default='0', help='GPU index, -1 for CPU')
parser.add_argument('--console', action='store_true', help="Optimize printout for console output; otherwise for file")
parser.add_argument('--seed', default=42, type=int, help='Seed used for splitting sets')

parser.add_argument('--contrastive_weight', type=float, default='0.2')
parser.add_argument('--MEE', type=float, default='0.2')
parser.add_argument('--sigma', type=float, default='1.0')

args = parser.parse_args()

def process_ecog_data(
    dataset_filenames,
    point_filenames,
    noise_channels,
    pre_time,
    post_time,
    neg_num_rate = 1.2,
    sample_rate = 500):
    """
    处理ECOG数据，返回数据、标签和删除噪音通道后的索引映射。

    参数：
    - dataset_filenames: 数据集文件名的列表，例如 ['org_bandpass_data.mat', 'S2_bandpass_data.mat', ...]
    - point_filenames: 对应的叫声点文件名列表，例如 ['org_point.txt', 'S2_point.txt', ...]
    - noise_channels: 要移除的噪音通道列表（1为起始索引），例如 [2, 4, 29, 31, 66, 68, 93, 95]
    - pre_time_ms: 鸣叫前的时间（毫秒），例如 3200
    - post_time_ms: 鸣叫后的时间（毫秒），例如 200
    - sample_rate: 采样率，默认500Hz

    返回：
    - data: 处理后的数据样本，numpy数组
    - labels: 数据对应的标签，numpy数组
    - index_mapping: 删除噪音通道后索引对应的字典
    """


    # 计算对应的采样点数
    pre_samples = int(pre_time * sample_rate)
    post_samples = int(post_time * sample_rate)
    total_samples = pre_samples - post_samples  # 总共的样本数

    # 将噪音通道转换为0基索引
    noise_channels_zero_based = [x - 1 for x in noise_channels]

    # 初始化
    sound_segments = []  # 正样本
    stimulus_negative_segments = []  # 刺激负样本
    index_mapping = {}  # 索引映射

    # 函数：移除噪声通道
    def remove_noise_channels(data, noise_channels):
        total_channels = data.shape[0]
        original_indices = np.arange(total_channels)  # 原始索引
        remaining_channels = np.delete(data, noise_channels, axis=0)  # 删除指定通道
        remaining_indices = np.delete(original_indices, noise_channels)  # 删除指定通道后的索引
        return remaining_channels, remaining_indices

    # 函数：根据时间点切分正样本
    def cut_segments(data, points):
        segments = []
        for point in tqdm(points, desc="切分正样本"):
            start_idx = int(point * sample_rate) - pre_samples
            end_idx = int(point * sample_rate) - post_samples
            if start_idx >= 0 and end_idx <= data.shape[1]:
                segment = data[:, start_idx:end_idx]
                segments.append(segment)
            else:
                print(f"警告: 时间点 {point} 超出范围，跳过。")
        return segments

     # 添加 tqdm 进度条到切分负样本
    def cut_stimulus_negative_segments(data, points):
        segments = []
        num_samples = data.shape[1]
        exclude_zone = int(10 * sample_rate)  # 排除每个叫声点前后10秒的区域
        segment_length = total_samples
        exclusion_zones = [(max(0, int(point * sample_rate) - exclude_zone), 
                            min(num_samples, int(point * sample_rate) + exclude_zone)) for point in points]

        start_idx = 0
        while start_idx + segment_length <= num_samples:
            valid = True
            for start_exclude, end_exclude in exclusion_zones:
                if start_idx < end_exclude and (start_idx + segment_length) > start_exclude:
                    start_idx = end_exclude
                    valid = False
                    break
            if valid:
                end_idx = start_idx + segment_length
                if end_idx <= num_samples:
                    segment = data[:, start_idx:end_idx]
                    segments.append(segment)
                start_idx += segment_length
            else:
                start_idx += segment_length  # 跳过排除区域
        return segments

    # 处理每个数据集
    for dataset_filename, point_filename in zip(dataset_filenames, point_filenames):
        # 加载数据集
        mat_data = sio.loadmat(dataset_filename)
        data_key = [key for key in mat_data.keys() if not key.startswith('__')][0]  # 获取数据变量名
        data = mat_data[data_key]

        # 移除噪音通道
        data_cleaned, remaining_indices = remove_noise_channels(data, noise_channels_zero_based)
        index_mapping[dataset_filename] = remaining_indices

        # 加载时间点
        with open(point_filename, 'r') as f:
            points = list(map(float, f.read().split(',')))

        # 正样本切分
        sound_segments.extend(cut_segments(data_cleaned, points))

        # 刺激负样本切分
        stimulus_negative_segments.extend(cut_stimulus_negative_segments(data_cleaned, points))
    
    # 将正负样本转换为 numpy 数组
    positive_data = np.array(sound_segments)
    negative_data = np.array(stimulus_negative_segments)

    # 创建标签
    positive_labels = np.ones(positive_data.shape[0])
    negative_labels = np.zeros(negative_data.shape[0])
    
    # 合并数据和标签
    data = np.vstack((positive_data, negative_data))
    labels = np.hstack((positive_labels, negative_labels))
    
    # 获取负样本（标签为0）的索引
    negative_indices = np.where(labels == 0)[0]

    # 设置随机种子保证结果可重复
    random.seed(42)

    # 随机选择neg_num_rate倍正样本的负样本
    neg_num = int(len(sound_segments)*neg_num_rate)
    selected_negative_indices = random.sample(list(negative_indices), neg_num)

    # 获取正样本（标签为1）的索引
    positive_indices = np.where(labels == 1)[0]

    # 合并所有正样本和220个随机选取的负样本
    selected_indices = np.concatenate((positive_indices, selected_negative_indices))

    # 根据选取的索引构建新的数据和标签
    new_data = data[selected_indices]
    new_labels = labels[selected_indices]
    
    # 保存为MAT文件
    output_filename = 'filtered_ecog_data.mat'
    def truncate_keys(d, maxlen=31):
        return {k[:maxlen]: v for k, v in d.items()}
    index_mapping = truncate_keys(index_mapping)
    sio.savemat(output_filename, {'data': new_data, 'labels': new_labels,'index_mapping':index_mapping})

    print(f"新的数据集已保存为 {output_filename}")
    print("新数据形状：", new_data.shape)
    print("新标签形状：", new_labels.shape)
    
    return new_data, new_labels, index_mapping


def split_dataset(data, labels, test_ratio=0.2, random_seed=42, stratify=False):
    np.random.seed(random_seed)
    num_samples = len(data)
    
    if stratify:
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, 
            test_size=test_ratio, 
            stratify=labels,
            random_state=random_seed
        )
    else:
        indices = np.random.permutation(num_samples)
        split_idx = int(num_samples * (1 - test_ratio))
        
        train_data, test_data = data[indices[:split_idx]], data[indices[split_idx:]]
        train_labels, test_labels = labels[indices[:split_idx]], labels[indices[split_idx:]]
    
    return (train_data, train_labels), (test_data, test_labels)


if __name__ == '__main__':
    print("======== 实验环境信息 ========")

    # 系统信息
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python 版本: {platform.python_version()}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    else:
        print("未检测到可用的 GPU")
    #print("======== 实验数据信息 ========")
    config = Setup(args)  # configuration dictionary
    random.seed(config['seed']) 
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])      # 为CPU设置随机种子
    torch.cuda.manual_seed(config['seed']) # 为当前GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # 让 CuDNN 选择确定性算法
    #torch.backends.cudnn.benchmark = True     # 关闭自动优化，保证复现性
    device = Initialization(config)
    list_dataset_name = os.listdir(config['data_path'])
    list_dataset_name.sort()
    for problem in list_dataset_name[config['dataset_pos']:config['dataset_pos'] + 1]:  # for loop on the all datasets in "data_dir" directory
        config['data_dir'] = config['data_path'] +"/"+ problem
        print(problem)
        # ------------------------------------ Load Data ---------------------------------------------------------------
        #logger.info("加载数据...")
        dataset_filenames = ['new_data/S2/bandpass_data.mat','new_data/S3/bandpass_data.mat','new_data/S4/bandpass_data.mat']
        point_filenames = ['new_data/S2/point_S2.txt','new_data/S3/point_S3.txt','new_data/S4/point_S4.txt']
        remove_indices = [1, 3, 5, 12, 18, 23, 28, 30, 40, 52, 57, 63, 65, 67, 92, 94, 99, 116]
        noise_channels = [x + 1 for x in remove_indices]
        #noise_channels = []
        pre_time = 3.3
        post_time = 0.3
        neg_num_rate = 1.2
        sample_rate = 500
        '''
        data,labels,index_mapping = process_ecog_data(dataset_filenames = dataset_filenames,
                                                    point_filenames = point_filenames,
                                                    noise_channels = noise_channels,
                                                    pre_time = pre_time,
                                                    post_time = post_time,
                                                    neg_num_rate = neg_num_rate,
                                                    sample_rate = sample_rate)
        '''
        # 读取MAT文件
        mat_data = sio.loadmat('filtered_ecog_data.mat')
        # 提取数据
        data = mat_data['data']                   # 读取 ECoG 数据
        labels = mat_data['labels'].squeeze()     # 读取标签
        index_mapping = mat_data['index_mapping']  # 读取索引映射
        (train_data, train_label), (val_data, val_label) = split_dataset(data, labels, test_ratio=0.2, stratify=False)
        (val_data, val_label), (test_data, test_label) = split_dataset(val_data, val_label, test_ratio=0.5, stratify=False)
        Data = {}
        pre_data = train_data[:config['num_train']]
        pre_label = train_label[:config['num_train']]
 
        Data['max_len'] = train_data.shape[2]
        Data['train_data'] = train_data
        Data['train_label'] = train_label
        Data['pre_data'] = pre_data
        Data['pre_label'] = pre_label
        Data['val_data'] = val_data
        Data['val_label'] = val_label
        Data['test_data'] = test_data
        Data['test_label'] = test_label


        len_ts = 1500
        dim = 110
  
        # --------------------------------------------------------------------------------------------------------------
        # -------------------------------------------- Shapelet Discovery ----------------------------------------------
        shapelet_discovery = ShapeletDiscover(window_size=args.window_size, num_pip=args.num_pip,
                                              processes=args.processes, len_of_ts=len_ts, dim=dim)
        sc_path = "store/" + problem + "_" + str(args.window_size) + ".pkl"
        if args.pre_shapelet_discovery == 1:
            shapelet_discovery.load_shapelet_candidates(path=sc_path)
        else:
            time_s = time.time()
            shapelet_discovery.extract_candidate(train_data=pre_data)
            shapelet_discovery.discovery(train_data=pre_data, train_labels=pre_label)
            shapelet_discovery.save_shapelet_candidates(path=sc_path)
            print("shapelet discovery time: %s" % (time.time() - time_s))

        shapelets_info = shapelet_discovery.get_shapelet_info(number_of_shapelet=args.num_shapelet)

        sw = torch.tensor(shapelets_info[:,3])
        sw = torch.softmax(sw.float()*20, dim=0)*sw.shape[0]
        shapelets_info[:,3] = sw.numpy()
        shapelets = []
        
        fs = 500  # 采样率 Hz
        max_freq = 8
        for si in shapelets_info:
            sc = pre_data[int(si[0]), int(si[5]), int(si[1]):int(si[2])]
            shapelets.append(sc)
            
        config['shapelets_info'] = shapelets_info
        config['shapelets'] = shapelets
        config['len_ts'] = len_ts
        config['ts_dim'] = dim

        train_dataset = dataset_class(Data['train_data'], Data['train_label'])
        val_dataset = dataset_class(Data['val_data'], Data['val_label'])
        test_dataset = dataset_class(Data['test_data'], Data['test_label'])
        g = torch.Generator()
        g.manual_seed(config['seed'])
        train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], num_workers=8, generator=g, shuffle=True, pin_memory=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], num_workers=8, generator=g, shuffle=True, pin_memory=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], num_workers=8, generator=g, shuffle=True, pin_memory=True)
        print("======== 脑电信号分割 ========")
        
        # --------------------------------------------------------------------------------------------------------------
        # -------------------------------------------- Build Model -----------------------------------------------------
        dic_position_results = [config['data_dir'].split('/')[-1]]

        #logger.info("Creating model ...")
        config['Data_shape'] = Data['train_data'].shape
        config['num_labels'] = int(max(Data['train_label']))+1
        #print("类别数:", config['num_labels'])
        model = model_factory(config)

        # -------------------------------------------- Model Initialization ------------------------------------
        optim_class = get_optimizer("Adam")
        config['optimizer'] = optim_class(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        config['loss_module'] = get_loss_module()
        save_path = os.path.join(config['save_dir'], problem + 'model_{}.pth'.format('last'))
        tensorboard_writer = SummaryWriter('summary')
        model.to(device)

        # ---------------------------------------------- Training The Model ------------------------------------
        logger.info('Starting training...')
        trainer = SupervisedTrainer(model, train_loader, device, config['loss_module'], config['optimizer'], l2_reg=0.0, contrastive_weight=config['contrastive_weight'], MEE_loss=config['MEE'], sigma=config['sigma'],
                                    print_interval=config['print_interval'], console=config['console'], print_conf_mat=False)
        test_evaluator = SupervisedTrainer(model, val_loader, device, config['loss_module'],
                                          print_interval=config['print_interval'], console=config['console'],
                                          print_conf_mat=False)

        train_runner(config, model, trainer, test_evaluator, save_path)

        best_model, optimizer, start_epoch = load_model(model, save_path, config['optimizer'])
        best_model.to(device)


        print("======== 实验结果展示 ========")

        test_loader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=8, generator=g, shuffle=True, pin_memory=True)
        best_test_evaluator = SupervisedTrainer(best_model, test_loader, device, config['loss_module'],
                                                print_interval=config['print_interval'], console=config['console'],
                                                print_conf_mat=False)
        best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)




        
        
    
    
        

        