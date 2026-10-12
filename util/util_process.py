import os
import numpy as np
import torch
from Bio import SeqIO
from torch.utils.data import TensorDataset, DataLoader


def load_fasta_seq(root_seq_folder, is_training=True):
    """
    读取蛋白质序列 fasta 文件

    返回：
        seq_dict: {protein_id: sequence}
        labels_dict: {protein_id: label}
    """

    pos_file = os.path.join(root_seq_folder,'train_pos.fasta' if is_training else 'test_pos.fasta')
    neg_file = os.path.join(root_seq_folder,'train_neg.fasta' if is_training else 'test_neg.fasta' )

    seq_dict = {}
    labels_dict = {}

    def process_sequences(sequences, label):
        for record in sequences:
            protein_id = str(record.id)
            seq = str(record.seq)

            seq_dict[protein_id] = seq
            labels_dict[protein_id] = label

    pos_sequences = SeqIO.parse(pos_file, "fasta")
    process_sequences(pos_sequences, 1)

    neg_sequences = SeqIO.parse(neg_file, "fasta")
    process_sequences(neg_sequences, 0)

    return seq_dict, labels_dict


def load_embeddings(root_npy_folder, max_length, is_training=True, embedding_type='ProtT5'):
    pos_folder = os.path.join(root_npy_folder, f'train_pos_{embedding_type}' if is_training else f'test_pos_{embedding_type}')
    neg_folder = os.path.join(root_npy_folder, f'train_neg_{embedding_type}' if is_training else f'test_neg_{embedding_type}')

    features_dict = {}
    labels_dict = {}

    def load_samples(folder_path, label):
        """加载指定文件夹中的样本"""
        for filename in os.listdir(folder_path):
            if filename.endswith('.npy'):
                protein_id = filename[:-4]
                feature = np.load(os.path.join(folder_path, filename))
                squeezed_feature = np.squeeze(feature)

                if squeezed_feature.shape[0] > max_length:
                    padded_feature = squeezed_feature[:max_length, :]
                else:
                    padding = np.zeros((max_length - squeezed_feature.shape[0], squeezed_feature.shape[1]))
                    padded_feature = np.vstack([squeezed_feature, padding])

                features_dict[protein_id] = padded_feature
                labels_dict[protein_id] = label

    load_samples(pos_folder, label=1)
    load_samples(neg_folder, label=0)

    return features_dict, labels_dict


def load_fasta_ss8(root_ss_folder, max_length=100, is_training=True):
    """
    读取二级结构 fasta 文件

    返回：
        ss_features_dict: {protein_id: padded_ss_array}
        labels_dict
        mask_dict: {protein_id: padding_mask_array}
    """
    pos_file = os.path.join(root_ss_folder,'train_pos_ss8.fasta' if is_training else 'test_pos_ss8.fasta' )
    neg_file = os.path.join(root_ss_folder,'train_neg_ss8.fasta' if is_training else 'test_neg_ss8.fasta'   )

    ss_features_dict = {}
    labels_dict = {}
    mask_dict = {}

    # H/E/C/S/T/G/I/B/X
    ss_map = {'H': 0, 'E': 1, 'C': 2, 'S': 3,'T': 4, 'G': 5, 'I': 6, 'B': 7, 'X': 8 }
    padding_id = ss_map['X']

    def process_sequences(sequences, label):
        for record in sequences:
            protein_id = str(record.id)
            ss_seq = str(record.seq)

            # 编码
            encoded = [ss_map.get(ch, padding_id) for ch in ss_seq]

            # 截断 or padding
            if len(encoded) > max_length:
                padded = encoded[:max_length]
            else:
                padded = encoded + [padding_id] * (max_length - len(encoded))

            padded = np.array(padded)

            # 生成 mask (True = padding)
            mask = (padded == padding_id)

            ss_features_dict[protein_id] = padded
            labels_dict[protein_id] = label
            mask_dict[protein_id] = mask.astype(bool)

    pos_sequences = SeqIO.parse(pos_file, "fasta")
    process_sequences(pos_sequences, 1)

    neg_sequences = SeqIO.parse(neg_file, "fasta")
    process_sequences(neg_sequences, 0)
        
    return ss_features_dict, labels_dict, mask_dict


batch_size = 64
def construct_dataset(protT5_dict, ss_dict, mask_dict, labels_dict, train=True, batch_size=batch_size):
    protT5_list = []
    ss_list = []
    mask_list = []
    labels_list = []

    for protein_id in protT5_dict.keys():
        protT5_list.append(protT5_dict[protein_id])
        ss_list.append(ss_dict[protein_id])
        mask_list.append(mask_dict[protein_id])
        labels_list.append(labels_dict[protein_id])

    protT5_tensor = torch.FloatTensor(np.array(protT5_list))
    ss_tensor = torch.LongTensor(np.array(ss_list))
    mask_tensor = torch.BoolTensor(np.array(mask_list))
    labels_tensor = torch.LongTensor(labels_list)

    dataset = TensorDataset(protT5_tensor, ss_tensor, mask_tensor, labels_tensor, )
    data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return data_iter
