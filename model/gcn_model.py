import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, InMemoryDataset  
from transformers import BertModel, BertTokenizer

from supar import Parser
import re

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, PLACE_HOLDER='ENTITY', device='cuda:1'):
        super(GCNEncoder, self).__init__()

        self.PLACE_HOLDER = PLACE_HOLDER
        self.device = device
        
        torch.cuda.set_device('cuda:0')  
        self.parser = Parser.load('/home/holo/.cache/supar/ptb.biaffine.dep.lstm.char')
        self.bert_model = BertModel.from_pretrained('/home/holo/data/scibert_scivocab_uncased').to(device)  # Move BERT to specified device
        # Freeze BERT parameters
        for param in self.bert_model.parameters():
            param.requires_grad = False
            
        self.tokenizer = BertTokenizer.from_pretrained('/home/holo/data/scibert_scivocab_uncased')

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, out_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(out_channels, out_channels))  # Update input size for subsequent layers

    def process(self, word_batch, edge_index_batch):
        data_list = []
        for words, edge_index in zip(word_batch, edge_index_batch):
            data = self.preprocess(words, edge_index)
            data_list.append(data)
        return data_list

    def align_outputs(self, words, tokens):
        alignment = []
        w_idx, t_idx = 0, 0
        words = [word.lower() for word in words]

        while w_idx < len(words):
            current_word = words[w_idx]
            current_token = tokens[t_idx]
            
            # 跳过[CLS]和[SEP]标记
            if current_token in ['[CLS]', '[SEP]']:
                t_idx += 1
                continue
            
            indices = []
            token_concat = ""
            
            # 拼接tokens直到匹配当前的word
            while current_word != (token_concat):
                indices.append(t_idx)
                if tokens[t_idx] not in ['[CLS]', '[SEP]']: 
                    token_concat += tokens[t_idx][2:] if tokens[t_idx].startswith("##") else tokens[t_idx]
                t_idx += 1

            alignment.append(indices)
            w_idx += 1

        return alignment

    def pool_tokens(self, hidden_states, indices, method='mean'):
        reprs = hidden_states[indices]
        if method == 'mean':
            return reprs.mean(dim=0)
        elif method == 'max':
            return reprs.max(dim=0)
        else:
            raise ValueError("Unsupported pooling method. Use 'mean' or 'max'.")

    def get_bert_outputs(self, words):
        all_tokens = []
        all_outputs = []

        # 初始输入
        text = ' '.join(words)
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # 当分词后的长度超过512时进行处理
        while len(tokens) > 512:
            cut_position = 509
            while cut_position > 0 and tokens[cut_position] != '.':
                cut_position -= 1

            # 如果没有找到句点，则切分至512个token
            if cut_position == 0:
                cut_position = 511

            # 提取当前 chunk 的文本
            chunk_text = self.tokenizer.decode(inputs['input_ids'][0][:cut_position + 1], skip_special_tokens=True)
            chunk_text = re.sub(r'\.(\s)', '. ', chunk_text)
            
            # 对 chunk 文本进行重新 tokenization 并确保不超过 BERT 的最大长度限制
            chunk_inputs = self.tokenizer(chunk_text, return_tensors='pt', truncation=True, max_length=512).to(self.device)

            # 计算当前 chunk 的 BERT 输出
            with torch.no_grad():
                chunk_output = self.bert_model(**chunk_inputs).last_hidden_state
                all_outputs.append(chunk_output)
                all_tokens.append(self.tokenizer.convert_ids_to_tokens(chunk_inputs['input_ids'][0]))

            # 更新文本，移除已处理部分
            text = self.tokenizer.decode(inputs['input_ids'][0][cut_position + 1:], skip_special_tokens=True)
            inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # 最后处理不足512长度的剩余部分
        if len(tokens) > 0:
            final_inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                final_output = self.bert_model(**final_inputs).last_hidden_state
                all_outputs.append(final_output)
                all_tokens.append(self.tokenizer.convert_ids_to_tokens(final_inputs['input_ids'][0]))

        # 将所有块的输出拼接
        all_outputs = torch.cat(all_outputs, dim=1).squeeze(0)
        all_tokens = [token for chunk_tokens in all_tokens for token in chunk_tokens]
        return all_tokens, all_outputs

    def preprocess(self, words, edge_index, method='mean'):
        # Get BERT outputs
        tokens, bert_outputs = self.get_bert_outputs(words)
        alignment = self.align_outputs(words, tokens)
        node_features = []
        for indices in alignment:
            feature = self.pool_tokens(bert_outputs, indices, method)
            node_features.append(feature)
        
        edge_index = edge_index.clone().detach().contiguous().to(self.device)
        graph_data = Data(x=torch.stack(node_features).to(self.device), edge_index=edge_index.to(self.device))  

        return graph_data
    
    def forward(self, word_batch, edge_index_batch):
        # Process text data
        data_list = self.process(word_batch, edge_index_batch)

        result = torch.tensor([]).to(self.device)

        for idx, data in enumerate(data_list):
            x, edge_index = data.x, data.edge_index 
            try:    
                for conv in self.convs:
                    x = conv(x, edge_index)
                    x = F.relu(x)  
                result = torch.cat((result, x[1].unsqueeze(0)), 0)
            except: 
                print("Error")
                print(x.shape)
                print(edge_index.shape)
                print(word_batch[idx])
                print(data)
                raise ValueError("Error")

        return result

class PubChemDataset(InMemoryDataset):
    def __init__(self, path):
        super(PubChemDataset, self).__init__()
        self.data, self.slices = torch.load(path)
    
    def __getitem__(self, idx):
        return self.get(idx)

if __name__ == "__main__":
    # 模型参数
    in_channels = 768
    out_channels = 256
    num_layers = 3
    model = GCNEncoder(in_channels, out_channels, num_layers, device="cuda:0").to("cuda:0")  # Move model to GPU
    dataset = PubChemDataset('./data/PubChem324kSP/train.pt')

    # 模拟输入
    words = dataset[0].words
    edge_index = dataset[0].text_edge_index

    # 前向传播测试
    output = model([words], [edge_index])
    # print(output)
    print(output)

    # print("Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # tokens, bert_outputs = model.get_bert_outputs(sample_texts[0])
    # print(tokens[511])
