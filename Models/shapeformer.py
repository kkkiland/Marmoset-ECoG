import numpy as np
from torch import nn
import torch
from Models.AbsolutePositionalEncoding import LearnablePositionalEncoding
from Models.Attention import Attention, ChannelAttention
from Models.position_shapelet import PPSN
from einops import rearrange
from torch.nn.functional import gumbel_softmax
from Models.cross_channel_Transformer import Trans_C

def create_patches(data, patch_size, stride):
    b, c, t = data.shape  # (batch, channels, time)
    num_patches = (t - patch_size) // stride + 1
    patches = []
    for i in range(num_patches):
        patch = data[:, :, i * stride : i * stride + patch_size]  # 取出时间窗口
        patches.append(patch)
    return torch.stack(patches, dim=1) 

class Permute(nn.Module):
    def forward(self, x):
        return x.permute(1, 0, 2)

class channel_mask_generator(torch.nn.Module):
    def __init__(self, input_size, n_vars):
        super(channel_mask_generator, self).__init__()
        self.generator = nn.Sequential(torch.nn.Linear(input_size, n_vars, bias=False), nn.Sigmoid())
        with torch.no_grad():
            self.generator[0].weight.zero_()
        self.n_vars = n_vars

    def forward(self, x):  # x: [(bs x patch_num) x n_vars x patch_size]

        distribution_matrix = self.generator(x)

        resample_matrix = self._bernoulli_gumbel_rsample(distribution_matrix)

        inverse_eye = 1 - torch.eye(self.n_vars).to(x.device)
        diag = torch.eye(self.n_vars).to(x.device)

        resample_matrix = torch.einsum("bcd,cd->bcd", resample_matrix, inverse_eye) + diag

        return resample_matrix

    def _bernoulli_gumbel_rsample(self, distribution_matrix):
        b, c, d = distribution_matrix.shape

        flatten_matrix = rearrange(distribution_matrix, 'b c d -> (b c d) 1')
        eps = 1e-5
        flatten_matrix = flatten_matrix.clamp(min=eps, max=1-eps)
        r_flatten_matrix = 1 - flatten_matrix
        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix)
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix)

        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        resample_matrix = gumbel_softmax(new_matrix, hard=True)

        # 获取采样结果（取第一个通道）
        resample_matrix = resample_matrix[:, 0]  
        # 恢复三维形状
        resample_matrix = resample_matrix.view(b, c, d) 
        
        return resample_matrix


def model_factory(config):
    model = Shapeformer(config, num_classes=config['num_labels'])
    return model

class ShapeBlock(nn.Module):
    def __init__(self, shapelet_info=None, shapelet=None, shape_embed_dim=32, window_size=100, len_ts=100, norm=1000, max_ci=3):
        super(ShapeBlock, self).__init__()
        self.dim = shapelet_info[5]
        self.shape_embed_dim = shape_embed_dim
        self.shapelet = torch.nn.Parameter(torch.tensor(shapelet, dtype=torch.float32), requires_grad=True)
        self.window_size = window_size
        self.norm = norm
        self.kernel_size = shapelet.shape[-1]
        self.weight = shapelet_info[3]

        self.ci_shapelet = np.sqrt(np.sum((shapelet[1:]- shapelet[:-1])**2)) + 1/norm
        self.max_ci = max_ci

        self.sp = shapelet_info[1]
        self.ep = shapelet_info[2]

        self.start_position = int(shapelet_info[1] - window_size)
        self.start_position = self.start_position if self.start_position >= 0 else 0
        self.end_position = int(shapelet_info[2] + window_size)
        self.end_position = self.end_position if self.end_position < len_ts else len_ts

        self.l1 = nn.Linear(self.kernel_size, shape_embed_dim)
        self.l2 = nn.Linear(self.kernel_size, shape_embed_dim)



    def forward(self, x):
        pis = x[:, self.dim, self.start_position:self.end_position]

        # 展开时间序列为长度为 kernel_size 的片段
        pis = pis.unfold(1, self.kernel_size, 1).contiguous()
        pis = pis.view(-1, self.kernel_size)

        # 计算每个片段与 shapelet 的皮尔逊相关系数
        shapelet_mean = self.shapelet.mean()
        shapelet_std = self.shapelet.std()
        pis_mean = pis.mean(dim=1, keepdim=True)
        pis_std = pis.std(dim=1, keepdim=True)
        covariance = ((pis - pis_mean) * (self.shapelet - shapelet_mean)).mean(dim=1)
        correlation = covariance / (pis_std.squeeze() * shapelet_std)

        # 选择与 shapelet 相关性最高的片段
        correlation = correlation.view(x.size(0), -1)
        index = torch.argmax(correlation, dim=1)
        pis = pis.view(x.size(0), -1, self.kernel_size)
        out = pis[torch.arange(x.size(0)), index]

        # 通过线性层
        out = self.l1(out)
        out_s = self.l2(self.shapelet.unsqueeze(0))

        # 计算输出
        out = out - out_s

        return out.view(x.shape[0], 1, -1)
        
class Shapeformer(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Shapelet Query  ---------------------------------------------------------
        self.shapelet_info = config['shapelets_info']
        self.shapelet_info = torch.IntTensor(self.shapelet_info)
        self.shapelets = config['shapelets']
        self.sw = torch.nn.Parameter(torch.tensor(config['shapelets_info'][:, 3]).float(), requires_grad=True)
        self.hidden_dim = 128
        # Local Information
        self.len_w = config['len_w']
        self.pad_w = self.len_w - config['len_ts'] % self.len_w
        self.pad_w = 0 if self.pad_w == self.len_w else self.pad_w
        self.height = config['ts_dim']
        self.weight = int(np.ceil(config['len_ts'] / self.len_w))

        list_d = []
        list_p = []
        for d in range(self.height):
            for p in range(self.weight):
                list_d.append(d)
                list_p.append(p)

        list_ed = position_embedding(torch.tensor(list_d))
        list_ep = position_embedding(torch.tensor(list_p))
        self.local_pos_embedding = torch.cat((list_ed, list_ep), dim=1)

        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        self.seq_len = seq_len
        self.channel_size = channel_size
        dim_ff = config['dim_ff']
        num_heads = config['num_heads']
        local_pos_dim = config['local_pos_dim']
        local_embed_dim = config['local_embed_dim']
        local_emb_size = local_embed_dim
        self.local_emb_size = local_emb_size
        localz_emb_size = 4

        self.local_layer = nn.Linear(self.len_w, local_embed_dim)
        self.embed_layer = nn.Sequential(nn.Conv2d(1, local_emb_size * 1, kernel_size=[1, 8], padding='same'),
                                         nn.BatchNorm2d(local_emb_size * 1),
                                         nn.GELU())
                                    

        self.embed_layer2 = nn.Sequential(
            #nn.Conv2d(local_emb_size * 1, local_emb_size, kernel_size=[self.hidden_dim * 2, 1], padding='valid'),
            nn.Conv2d(local_emb_size * 1, local_emb_size, kernel_size=[channel_size, 1], padding='valid'),
            nn.BatchNorm2d(local_emb_size),
            nn.GELU())

        self.embed_layer3 = nn.Sequential(nn.Conv2d(1, localz_emb_size * 1, kernel_size=[1, 8], padding='same'),
                                         nn.BatchNorm2d(localz_emb_size * 1),
                                         nn.GELU())
                                    

        self.embed_layer4 = nn.Sequential(
            #nn.Conv2d(local_emb_size * 1, local_emb_size, kernel_size=[self.hidden_dim * 2, 1], padding='valid'),
            nn.Conv2d(localz_emb_size * 1, localz_emb_size, kernel_size=[channel_size, 1], padding='valid'),
            nn.BatchNorm2d(localz_emb_size),
            nn.GELU())
        self.Fix_Position = LearnablePositionalEncoding(local_emb_size, dropout=config['dropout'], max_len=seq_len)
        self.Fix_Position2 = LearnablePositionalEncoding(localz_emb_size, dropout=config['dropout'], max_len=seq_len)
        self.local_pos_layer = nn.Linear(self.local_pos_embedding.shape[-1], local_pos_dim)
        self.local_ln1 = nn.LayerNorm(local_emb_size, eps=1e-5)
        self.local_ln2 = nn.LayerNorm(local_emb_size, eps=1e-5)
        self.local_ln3 = nn.LayerNorm(localz_emb_size, eps=1e-5)
        self.local_ln4 = nn.LayerNorm(localz_emb_size, eps=1e-5)
        self.local_attention_layer = Attention(local_emb_size, num_heads, config['dropout'])
        self.local_attention_layer2 = Attention(localz_emb_size, 1, config['dropout'])
        self.local_ff = nn.Sequential(
            nn.Linear(local_emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, local_emb_size),
            nn.Dropout(config['dropout']))
        self.local_gap = nn.AdaptiveAvgPool1d(1)
        self.local_flatten = nn.Flatten()

        self.local_ff2 = nn.Sequential(
            nn.Linear(localz_emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, localz_emb_size),
            nn.Dropout(config['dropout']))
        self.local_gap2 = nn.AdaptiveAvgPool1d(1)
        self.local_flatten2 = nn.Flatten()
        
        # Global Information
        self.shape_blocks = nn.ModuleList([
            ShapeBlock(shapelet_info=self.shapelet_info[i], shapelet=self.shapelets[i],
                       shape_embed_dim=config['shape_embed_dim'], len_ts=config["len_ts"])
            for i in range(len(self.shapelet_info))])

        self.shapelet_info = config['shapelets_info']
        self.shapelet_info = torch.FloatTensor(self.shapelet_info)
        self.position = torch.index_select(self.shapelet_info, 1, torch.tensor([5, 1, 2]))
        # 1hot pos embedding
        self.d_position = self.position_embedding(self.position[:, 0])
        self.s_position = self.position_embedding(self.position[:, 1])
        self.e_position = self.position_embedding(self.position[:, 2])

        self.d_pos_embedding = nn.Linear(self.d_position.shape[1], config['pos_embed_dim'])
        self.s_pos_embedding = nn.Linear(self.s_position.shape[1], config['pos_embed_dim'])
        self.e_pos_embedding = nn.Linear(self.e_position.shape[1], config['pos_embed_dim'])

        # Parameters Initialization -----------------------------------------------
        emb_size = config['shape_embed_dim']

        self.LayerNorm1 = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)
        self.attention_layer = Attention(emb_size, num_heads, config['dropout'])

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size + local_emb_size + localz_emb_size, num_classes)
        self.att = ChannelAttention(channel_size)
        # Merge Layer----------------------------------------------------------
        self.local_merge = nn.Linear(local_emb_size, int(local_emb_size / 2))
        self.patch_size = 50
        self.freq_transformer = Trans_C(dim=64, depth=3, heads=2,
                                       mlp_dim=256,
                                       dim_head=64, dropout=0.2,
                                       patch_dim=self.patch_size,
                                       horizon=192 * 2, d_model=128 * 2,
                                       regular_lambda=0.5, temperature=0.07)

        
        self.mask_generator_z = channel_mask_generator(input_size=self.patch_size, n_vars=channel_size)
        self.get = nn.Linear(128 * 2, 128 * 2)

    def position_embedding(self, position_list):
        max_d = position_list.max() + 1
        identity_matrix = torch.eye(int(max_d))
        d_position = identity_matrix[position_list.to(dtype=torch.long)]
        return d_position

    def forward(self, x, ep):
        local_x = x
        channel_size = self.channel_size
        patch_size = self.patch_size
        stride = 10        
        local_x = torch.fft.fft(local_x).real
        z = create_patches(local_x, patch_size, stride)  # (batch, num_patches, channels, patch_size)
        batch_size = z.shape[0]
        patch_num = z.shape[1]
        z = torch.reshape(z, (z.shape[0] * z.shape[1], z.shape[2], z.shape[-1]))  # z: [bs * patch_num,nvars, patch_size]
        
        channel_mask_z = self.mask_generator_z(z)
        #channel_mask_z = torch.eye(110, device=z.device).unsqueeze(0).repeat(z.shape[0], 1, 1)
        #channel_mask_z = torch.ones(z.shape[0], 110, 110).cuda()

        z, contrastive_loss, cosine_matrix = self.freq_transformer(z, channel_mask_z)

        z = self.get(z)
        z = torch.reshape(z, (batch_size, patch_num, channel_size, z.shape[-1]))
        z = z.permute(0, 2, 1, 3)  # z1: [bs, nvars, patch_num, horizon]
        z = torch.reshape(z, (batch_size, channel_size, -1))

        z = z[:,:,:self.seq_len]
        
        local_x = x.unsqueeze(1)
        local_x = self.embed_layer(local_x)
        local_x = self.embed_layer2(local_x).squeeze(2)
        local_x = local_x.permute(0, 2, 1)
        x_src_pos = self.Fix_Position(local_x)
        local_att = local_x + self.local_attention_layer(x_src_pos)
        local_att = self.local_ln1(local_att)
        local_out = local_att + self.local_ff(local_att)
        local_out = self.local_ln2(local_out)
        local_out = local_out.permute(0, 2, 1)
        local_out = self.local_gap(local_out)
        local_out = self.local_flatten(local_out)
        
        z = z.unsqueeze(1)
        local_z = self.embed_layer3(z)
        local_z = self.embed_layer4(local_z).squeeze(2)
        local_z = local_z.permute(0, 2, 1)
        z_src_pos = self.Fix_Position2(local_z)
        local_attz = local_z + self.local_attention_layer2(z_src_pos)
        local_attz = self.local_ln3(local_attz)
        local_outz = local_attz + self.local_ff2(local_attz)
        local_outz = self.local_ln4(local_outz)
        local_outz = local_outz.permute(0, 2, 1)
        local_outz = self.local_gap2(local_outz)
        local_outz = self.local_flatten2(local_outz)

        # Global information
        global_x = None
        for block in self.shape_blocks:
            if global_x is None:
                global_x = block(x)
            else:
                global_x = torch.cat((global_x, block(x)), dim=1)
        if self.d_position.device != x.device:
            self.d_position = self.d_position.to(x.device)
            self.s_position = self.s_position.to(x.device)
            self.e_position = self.e_position.to(x.device)

        d_pos = self.d_position.repeat(x.shape[0], 1, 1)
        s_pos = self.s_position.repeat(x.shape[0], 1, 1)
        e_pos = self.e_position.repeat(x.shape[0], 1, 1)

        d_pos_emb = self.d_pos_embedding(d_pos)
        s_pos_emb = self.s_pos_embedding(s_pos)
        e_pos_emb = self.e_pos_embedding(e_pos)



        global_x = global_x + d_pos_emb + s_pos_emb + e_pos_emb
        global_att = global_x + self.attention_layer(global_x)
        global_att = global_att * self.sw.unsqueeze(0).unsqueeze(2)
        global_att = self.LayerNorm1(global_att) # Choosing LN and BN
        global_out = global_att + self.FeedForward(global_att)
        global_out = self.LayerNorm2(global_out) # Choosing LN and BN
        global_out = global_out * self.sw.unsqueeze(0).unsqueeze(2)
        global_out, _ = torch.max(global_out, dim=1)

        
        hid = torch.cat((global_out,local_out,local_outz), dim=1)
        out = self.out(hid)
        return out, contrastive_loss, cosine_matrix




def position_embedding(position_list):
    max_d = position_list.max() + 1
    identity_matrix = torch.eye(int(max_d))
    d_position = identity_matrix[position_list.to(dtype=torch.long)]
    return d_position


if __name__ == '__main__':
    print()