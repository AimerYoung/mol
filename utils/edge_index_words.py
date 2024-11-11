"""数据清洗"""
import supar
import torch
import unicodedata
from transformers import BertModel, BertTokenizer
import re
from chemdataextractor import Document
from collections import defaultdict
from tqdm import tqdm

def load_data(file_path):
    """加载数据"""
    data = torch.load(file_path)
    return data[0], data[1]

def remove_accents(input_str):
    # 使用NFKD形式规范化字符串，这会将字符分解为其组成部分
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    # 过滤掉所有组合字符（即重音符号等），只保留非组合字符
    without_accents = ''.join([c for c in nfkd_form if not unicodedata.combining(c)])
    
    # 定义一个正则表达式来匹配不可见字符，包括零宽度字符、软连字符等
    invisible_chars = re.compile(
        r'[\u200B-\u200D\uFEFF\u00AD]'  # 零宽度字符，软连字符等
    )
    
    # 去除这些不可见字符
    clean_str = invisible_chars.sub('', without_accents)

    # 将句号后的空格替换为句号和空格
    text = re.sub(r'\.(\s)', '. ', clean_str)
    return text

def parse_iupac(text, PLACE_HOLDER='ENTITY'):
    doc = Document(text)
    iupac_names = [entity.text for entity in doc.cems if '-' in entity.text or '(' in entity.text]
    iupac_names = sorted(iupac_names, key=lambda x: len(x), reverse=True)
    replacements = []
    normal_text = text
    for iupac in iupac_names:
        start_idx = normal_text.find(iupac)
        if start_idx == -1:
            normal_text = normal_text.replace(iupac, PLACE_HOLDER, 1)
            replacements.append((len(replacements), iupac))

    return normal_text

def sep(text, parser):
    """Semantic dependency parsing"""
    dataset = parser.predict(text, lang='en', prob=True, verbose=False)
    return list(dataset.words[0]), dataset.arcs[0], dataset.rels[0], dataset.probs[0]

def pool_tokens(hidden_states, indices, method='mean'):
    reprs = hidden_states[indices]
    if method == 'mean':
        return reprs.mean(dim=0)
    elif method == 'max':
        return reprs.max(dim=0)
    else:
        raise ValueError("Unsupported pooling method. Use 'mean' or 'max'.")

def get_bert_outputs(words, bert_model, tokenizer, device="cuda"):
    all_tokens = []
    all_outputs = []

    # 初始输入
    text = ' '.join(words)
    inputs = tokenizer(text, return_tensors='pt').to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # 当分词后的长度超过512时进行处理
    while len(tokens) > 512:
        cut_position = 509
        while cut_position > 0 and tokens[cut_position] != '.':
            cut_position -= 1

        # 如果没有找到句点，则切分至512个token
        if cut_position == 0:
            cut_position = 511

        # 提取当前 chunk 的文本
        chunk_text = tokenizer.decode(inputs['input_ids'][0][:cut_position + 1], skip_special_tokens=True)
        chunk_text = re.sub(r'\.(\s)', '. ', chunk_text)
        
        # 对 chunk 文本进行重新 tokenization 并确保不超过 BERT 的最大长度限制
        chunk_inputs = tokenizer(chunk_text, return_tensors='pt', truncation=True, max_length=512).to(device)

        # 计算当前 chunk 的 BERT 输出
        with torch.no_grad():
            chunk_output = bert_model(**chunk_inputs).last_hidden_state
            all_outputs.append(chunk_output)
            all_tokens.append(tokenizer.convert_ids_to_tokens(chunk_inputs['input_ids'][0]))

        # 更新文本，移除已处理部分
        text = tokenizer.decode(inputs['input_ids'][0][cut_position + 1:], skip_special_tokens=True)
        inputs = tokenizer(text, return_tensors='pt').to(device)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # 最后处理不足512长度的剩余部分
    if len(tokens) > 0:
        final_inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            final_output = bert_model(**final_inputs).last_hidden_state
            all_outputs.append(final_output)
            all_tokens.append(tokenizer.convert_ids_to_tokens(final_inputs['input_ids'][0]))

    # 将所有块的输出拼接
    all_outputs = torch.cat(all_outputs, dim=1).squeeze(0)
    all_tokens = [token for chunk_tokens in all_tokens for token in chunk_tokens]
    return all_tokens, all_outputs

def align_outputs(words, tokens):
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

def process(text, parser, bert_model, bert_tokenizer):
    # 文本清洗
    text = remove_accents(text)
    # 化学实体识别
    normal_text = parse_iupac(text)
    # 依存分析
    words, arcs, rels, probs = sep(normal_text, parser)
    # 转换为邻接矩阵
    edge_index = [[i, arc - 1] for i, arc in enumerate(arcs) if arc != 0]
    edge_index = list(zip(*edge_index))
    
    return words, edge_index

def concatenate_data(processed_data, data, slices):
    words_data = [entry['words'] for entry in processed_data]
    data['words'] = words_data  # Keep it as a list of strings
    slices['words'] = torch.tensor([0] + [len(entry['words']) for entry in processed_data], dtype=torch.long).cumsum(dim=0)[:-1]
    
    text_edge_index = [torch.tensor(entry['text_edge_index'], dtype=torch.long) for entry in processed_data]
    data['text_edge_index'] = torch.cat(text_edge_index, dim=1)
    text_edge_index_lengths = [entry.shape[1] for entry in text_edge_index] 
    slices['text_edge_index'] = torch.tensor([0] + text_edge_index_lengths, dtype=torch.long).cumsum(dim=0)[:-1]
    slices['text_edge_index'] = torch.cat((slices['text_edge_index'], torch.tensor([data['text_edge_index'].shape[1]])))

    return data, slices


def save_data(data, file_path):
    """保存数据"""
    torch.save(data, file_path)

def main(input_path, output_path, parser, bert_model, bert_tokenizer):
    # Load and process data
    data, slice = load_data(input_path)
    text_data= data.text
    processed_data = []

    for i, text in enumerate(tqdm(text_data, desc="Processing data")):
        try:
            if i % 100 == 0:
                print(f"Processing entry {i+1}/{len(text_data)}...")
            words, edge_index = process(text, parser, bert_model, bert_tokenizer)
            processed_entry = {
                'words': words,
                'text_edge_index': edge_index,
            }
            processed_data.append(processed_entry)
        except Exception as e:
            print(f"Error processing entry {i+1}/{len(text_data)}: {e}")
            save_data(processed_data, f"{output_path}.partial")
            raise e

    # Concatenate data and create slices
    data, slices = concatenate_data(processed_data, data, slice)
    
    # Save the combined data and slices
    torch.save((data, slices), output_path)
    print(f"Data saved to {output_path}")


if __name__ == '__main__':
    parser = supar.Parser.load('/home/holo/.cache/supar/ptb.biaffine.dep.lstm.char')
    input_path = '/home/holo/Documents/mol/MolCA/data/PubChem324kV2/test.pt'
    output_path = '/home/holo/Documents/mol/spmol/data/PubChem324kSP/test1.pt'
    bert_model = BertModel.from_pretrained('/home/holo/data/scibert_scivocab_uncased').to('cuda')
    bert_tokenizer = BertTokenizer.from_pretrained('/home/holo/data/scibert_scivocab_uncased')
    main(input_path, output_path, parser, bert_model, bert_tokenizer)
    # load_data(input_path)