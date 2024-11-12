import torch
from torch_geometric.data import Data, Dataset
import torch_geometric
import sys
sys.path.append('/home/zhongcl/momu_v2/Pretrain/')
from utils.GraphAug import drop_nodes, permute_edges, subgraph, mask_nodes
from copy import deepcopy
import numpy as np
import os
import random
import json
import re
from transformers import BertTokenizer, BertModel, BertTokenizerFast
from torch_geometric.data import InMemoryDataset

class PubChemDataset(InMemoryDataset):
    def __init__(self, path):
        super(PubChemDataset, self).__init__()
        self.data, self.slices = torch.load(path)
    
    def __getitem__(self, idx):
        return self.get(idx)
    
class GINPretrainDataset(Dataset):
    def __init__(self, root, text_max_len):
        super(GINPretrainDataset, self).__init__(root)
        self.root = root
        self.text_max_len = text_max_len
        self.tokenizer = BertTokenizerFast.from_pretrained('/home/zhongcl/momu_v2/Pretrain/bert_pretrained')
        self.model = BertModel.from_pretrained('/home/zhongcl/momu_v2/Pretrain/bert_pretrained')
        self.rels = json.load(open('/home/zhongcl/M_T_Data/PubChem324kV2/all_rels.json','r'))

        graph_path = root + 'pretrain.pt'
        self.graph_datas = PubChemDataset(graph_path)
        text_file = open('/home/zhongcl/M_T_Data/PubChem324kV2/new_supar_result.json','r')
        self.text_datas = json.load(text_file)
        # self.graph_name_list = os.listdir(root+'graph/')
        # self.graph_name_list.sort()
        # self.text_name_list = os.listdir(root+'text_graph/')
        # self.text_name_list.sort()
        
    def __len__(self):
        # return len(self.graph_name_list)
        # return len(self.text_datas)
        return 20000

    def __getitem__(self, index):
        # graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]
        data = self.graph_datas[index]
        # data = torch.load(graph_path)[index]
        text_data = self.text_datas[index]

        x = data['x']
        edge_index = data['edge_index']
        edge_attr = data['edge_attr']

        smiles = data['smiles']

        words = text_data['words']
        sentence = text_data['sentence']
        # text_x = self.words_to_embeddings(words, sentence)
        arcs = text_data['arcs']
        text_edge_index = self.get_edge_index(arcs)
        text_rels = text_data['rels']
        rels = self.rels_to_id(text_rels)

        data_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        text_graph = Data(words=words, sentence=sentence, edge_index=text_edge_index, rels=rels)

        smiles_ids, smiles_attention_mask = self.tokenizer_func(smiles)

        # load and process graph
        # graph_path = os.path.join(self.root, 'graph', graph_name)
        # data_graph = torch.load(graph_path)

        # load and process text
        # text_path = os.path.join(self.root, 'text_graph', text_name)
        # text_graph = torch.load(text_path)
        return data_graph, text_graph, smiles_ids.squeeze(0), smiles_attention_mask.squeeze(0)
    
    def rels_to_id(self, rels):
        id = []
        for rel in rels:
            rel_id = self.rels[rel]
            id.append(int(rel_id))
        return torch.tensor(id)
    
    # def get_offsets(self, tokenized_words, text):
    #     offsets = []
    #     start_idx = 0  # 用于记录每个单词的起始位置
        
    #     for word in tokenized_words:
    #         start_idx = text.find(word, start_idx)  # 查找单词在文本中的位置
    #         end_idx = start_idx + len(word)  # 计算单词的结束位置
    #         offsets.append((word, start_idx, end_idx))  # 保存结果 (单词, 开始位置, 结束位置)
    #         start_idx = end_idx  # 更新起始位置以继续查找下一个单词
        
    #     return offsets
    
    def get_edge_index(self, arcs):
        # 创建文本边矩阵
        edge_index_1 = []
        edge_index_2 = []
        for i,arc in enumerate(arcs):
            if arc != 0:
                edge_index_1.append(i)
                edge_index_2.append(arc-1)
        text_edge_index = [edge_index_1, edge_index_2]
        text_edge_index = torch.tensor(text_edge_index, dtype=torch.long) 
        return text_edge_index


    # def words_to_embeddings(self, words, sentence):
    #     # all_embeddings = []
    #     # for word, sentence in zip(words, sentences):
    #     # with torch.no_grad():
    #         words_offsets = self.get_offsets(words,sentence)
    #         inputs = self.tokenizer(sentence, return_tensors='pt', return_offsets_mapping=True)
    #         offset_mapping = inputs['offset_mapping']
    #         # # 模型只需要 input_ids 和 attention_mask，不需要 offset_mapping
    #         model_inputs = {key: value for key, value in inputs.items() if key != 'offset_mapping'}
    #         outputs = self.model(**model_inputs)
    #         token_embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_size)

    #         word_embeddings = []           # 保存每个单词的最终嵌入
    #         current_word_embedding = []    # 临时保存当前单词的所有子词嵌入

    #         i = 0
    #         for idx, (embedding, offset) in enumerate(zip(token_embeddings, offset_mapping[0])):
    #             if offset[0] != 0 or offset[1] != 0:  # 开始[CLS]
    #                 # word = words_offsets[i][0]
    #                 start_id = words_offsets[i][1]
    #                 end_id = words_offsets[i][2]
    #                 if offset[0] == start_id and offset[1] == end_id:   # 完整的单词
    #                     word_embeddings.append(embedding)
    #                     i += 1
    #                 else:
    #                     current_word_embedding.append(embedding)
    #                     if offset[1] == end_id:
    #                         # 对单词的所有子词嵌入取平均
    #                         word_embeddings.append(torch.mean(torch.stack(current_word_embedding), dim=0))
    #                         current_word_embedding = []
    #                         i += 1
    #         # print(word_embeddings)
    #         word_embeddings = torch.stack(word_embeddings)
    #         # all_embeddings.append(word_embeddings)
    #         # final_embeddings = torch.tensor(all_embeddings)
    #         return word_embeddings


    def augment(self, data, graph_aug):
        # node_num = data.edge_index.max()
        # sl = torch.tensor([[n, n] for n in range(node_num)]).t()
        # data.edge_index = torch.cat((data.edge_index, sl), dim=1)
        if graph_aug == 'dnodes':
            data_aug = drop_nodes(deepcopy(data))
        elif graph_aug == 'pedges':
            data_aug = permute_edges(deepcopy(data))
        elif graph_aug == 'subgraph':
            data_aug = subgraph(deepcopy(data))
        elif graph_aug == 'mask_nodes':
            data_aug = mask_nodes(deepcopy(data))
        elif graph_aug == 'random2':  # choose one from two augmentations
            n = np.random.randint(2)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = subgraph(deepcopy(data))
            else:
                print('sample error')
                assert False
        elif graph_aug == 'random3':  # choose one from three augmentations
            n = np.random.randint(3)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = permute_edges(deepcopy(data))
            elif n == 2:
                data_aug = subgraph(deepcopy(data))
            else:
                print('sample error')
                assert False
        elif graph_aug == 'random4':  # choose one from four augmentations
            n = np.random.randint(4)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = permute_edges(deepcopy(data))
            elif n == 2:
                data_aug = subgraph(deepcopy(data))
            elif n == 3:
                data_aug = mask_nodes(deepcopy(data))
            else:
                print('sample error')
                assert False
        else:
            data_aug = deepcopy(data)
            data_aug.x = torch.ones((data.edge_index.max()+1, 1))

        return data_aug

    def tokenizer_func(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=False,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask


if __name__ == '__main__':
    bert_model = BertModel.from_pretrained('/home/zhongcl/momu_v2/Pretrain/bert_pretrained/')
    mydataset = GINPretrainDataset(root='/home/zhongcl/M_T_Data/PubChem324kV2/', text_max_len=256)
    # ids, attention_mask = mydataset.tokenizer_func(text='The molecule is a N-acyl-hexosamine')
    # print(ids)
    # print(attention_mask)
    sentence = "The molecule is an O-acylcarnitine having acetyl as the acyl substituent. It has a role as a human metabolite. It is functionally related to an acetic acid. It is a conjugate base of an O-acetylcarnitinium. The molecule is a natural product found in Pseudo-nitzschia multistriata, Euglena gracilis, and other organisms with data available. The molecule is a metabolite found in or produced by Saccharomyces cerevisiae. An acetic acid ester of CARNITINE that facilitates movement of ACETYL COA into the matrices of mammalian MITOCHONDRIA during the oxidation of FATTY ACIDS."

    res = mydataset.tokenizer.tokenize(sentence)
    ids,attention_mask = mydataset.tokenizer_func(sentence)
    # res = ['CLS'] + res + ['SEP']
    # print(res)
    # print(ids)
    # print(attention_mask)
    print(mydataset.tokenizer.token_to_ids('SEP'))

    # mydataset = GraphTextDataset()
    # train_loader = torch_geometric.loader.DataLoader(
    #     mydataset,
    #     batch_size=16,
    #     shuffle=True,
    #     num_workers=4
    # )
    # for i, (graph, text, smiles, mask) in enumerate(train_loader):
    #     print(graph.x.shape)
    #     print(graph.edge_index.shape)
    #     print(text.x.shape)
    #     print(text.edge_index.shape)
    #     print(text.shape)
    #     print(smiles.shape)
    #     print(mask.shape)
    # mydataset = GraphormerPretrainDataset(root='data/', text_max_len=128, graph_aug1='dnodes', graph_aug2='subgraph')
    # from functools import partial
    # from data_provider.collator import collator_text
    # train_loader = torch.utils.data.DataLoader(
    #         mydataset,
    #         batch_size=8,
    #         num_workers=4,
    #         collate_fn=partial(collator_text,
    #                            max_node=128,
    #                            multi_hop_max_dist=5,
    #                            spatial_pos_max=1024),
    #     )
    # graph, text1, mask1, text2, mask2 = mydataset[0]
    # mydataset = GINPretrainDataset(root='data/', text_max_len=128, graph_aug1='dnodes', graph_aug2='subgraph')
    # train_loader = torch_geometric.loader.DataLoader(
    #         mydataset,
    #         batch_size=32,
    #         shuffle=True,
    #         num_workers=0,
    #         pin_memory=False,
    #         drop_last=True,
    #         # persistent_workers = True
    #     )

    # for i, (graph, text1, mask1, text2, mask2) in enumerate(train_loader):
    #     print(graph)
        # print(graph.x.shape)
        # print(graph)
        # print(graph.x.dtype)
        # print(text1.shape)
        # print(mask1.shape)
        # print(text2.shape)
        # print(mask2.shape)