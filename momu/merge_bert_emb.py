from transformers import BertTokenizerFast, BertModel
import torch
import re

def reconstruct_text(tokenized_words):
    reconstructed_text = ""
    
    for i, word in enumerate(tokenized_words):
        # 如果是标点符号，紧跟在前一个单词后面
        if re.match(r'[^\w\s]', word):  # 标点符号
            reconstructed_text = reconstructed_text.rstrip() + word
            # 如果当前标点符号不是句子的结尾，则加一个空格
            if i < len(tokenized_words) - 1 and not re.match(r'[^\w\s]', tokenized_words[i+1]):
                reconstructed_text += " "
        else:
            # 不是标点符号，正常处理并在单词间加上空格
            if i > 0 and not re.match(r'[^\w\s]', tokenized_words[i-1]):
                reconstructed_text += " "
            reconstructed_text += word
    
    return reconstructed_text


def get_offsets(text, tokenized_words):
    offsets = []
    start_idx = 0  # 用于记录每个单词的起始位置
    
    for word in tokenized_words:
        start_idx = text.find(word, start_idx)  # 查找单词在文本中的位置
        end_idx = start_idx + len(word)  # 计算单词的结束位置
        offsets.append((word, start_idx, end_idx))  # 保存结果 (单词, 开始位置, 结束位置)
        start_idx = end_idx  # 更新起始位置以继续查找下一个单词
    
    return offsets

# 给定的文本和分词后的结果
text = "She is playing football."
tokenized_words = ['Hello', '!', 'She', 'is', 'playing', 'football', '.']
print(reconstruct_text(tokenized_words))

breakpoint()

# 获取每个单词的offset
words_offsets = get_offsets(text, tokenized_words)

# 输出结果
for word, start, end in words_offsets:
    print(f"'{word}': [{start}, {end}]")


tokenizer = BertTokenizerFast.from_pretrained('/home/zhongcl/momu_v2/Pretrain/bert_pretrained')
model = BertModel.from_pretrained('/home/zhongcl/momu_v2/Pretrain/bert_pretrained')

inputs = tokenizer(text, return_tensors='pt', return_offsets_mapping=True)

tokens = tokenizer.tokenize(text)
input_ids = inputs['input_ids']
offset_mapping = inputs['offset_mapping']
# ids_1 = tokenizer.convert_tokens_to_ids('[CLS]')
# ids_2 = tokenizer.convert_tokens_to_ids('[SEP]')
print(f'tokens: {tokens}')
print(f'input_ids: {input_ids}')
print(f'input_mappings: {offset_mapping}')

# # 模型只需要 input_ids 和 attention_mask，不需要 offset_mapping
model_inputs = {key: value for key, value in inputs.items() if key != 'offset_mapping'}
outputs = model(**model_inputs)
token_embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_size)
print(token_embeddings.shape)

word_embeddings = []           # 保存每个单词的最终嵌入
current_word_embedding = []    # 临时保存当前单词的所有子词嵌入

i = 0
for idx, (embedding, offset) in enumerate(zip(token_embeddings, offset_mapping[0])):
    if offset[0] != 0 or offset[1] != 0:  # 开始[CLS]
        word = words_offsets[i][0]
        start_id = words_offsets[i][1]
        end_id = words_offsets[i][2]
        if offset[0] == start_id and offset[1] == end_id:   # 完整的单词
            word_embeddings.append(embedding)
            i += 1
        else:
            current_word_embedding.append(embedding)
            if offset[1] == end_id:
                # 对单词的所有子词嵌入取平均
                word_embeddings.append(torch.mean(torch.stack(current_word_embedding), dim=0))
                current_word_embedding = []
                i += 1

print(word_embeddings)
word_embeddings = torch.stack(word_embeddings)
print(word_embeddings.shape)
