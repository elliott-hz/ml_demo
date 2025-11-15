# seq2seq_lstm_en_cmn_fixed.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ------------------------------
# Step 0: download data from https://www.manythings.org/anki/
# ------------------------------

# ------------------------------
# 配置参数
# ------------------------------
# 数据路径：manythings.org提供的中文数据集 (格式为: 英文 \t 中文 \t ...)
data_path = "cmn.txt"  
# 使用的样本数量（用于短实验，增加到2000以便模型能学到更多模式）
num_samples = 5000
# LSTM隐藏层维度：增加到256做一次超参实验（更大容量）
latent_dim = 256
# 词嵌入维度
embedding_dim = 128
# 批次大小
batch_size = 64
# 训练轮数（超参实验，较短）
epochs = 12
# Dropout比例（在小数据上降低以帮助学习）
dropout_rate = 0.1
# 学习率
learning_rate = 0.001

# ------------------------------
# 1. 加载和预处理数据
# ------------------------------
# 存储英文句子和中文句子的列表
input_texts = []
target_texts = []

# 检查数据文件是否存在
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}. Download and place 'cmn.txt' here.")

# 定义开始和结束标记
sos_token = "<sos>"  # Start of Sentence 标记
eos_token = "<eos>"  # End of Sentence 标记

print("Loading and preprocessing data...")

# 逐行读取数据文件
with open(data_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        # 只读取指定数量的样本
        if i >= num_samples:
            break
        # 分割英文和中文句子（通过制表符）
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        eng, cmn = parts[0], parts[1]
        # 转换为小写（仅对英文）
        eng = eng.lower()
        # 清理中文文本中的特殊字符
        cmn = cmn.replace('', '')  # 移除特殊字符
        # 中文保持原样

        # 在中文句子前后添加开始和结束标记
        # For Chinese we need to tokenize at the character level so the Tokenizer
        # doesn't treat each whole sentence as a single token. Insert spaces
        # between Chinese characters before adding <sos>/<eos>.
        cmn_clean = cmn.replace(' ', '')
        target = sos_token + " " + " ".join(list(cmn_clean)) + " " + eos_token

        # 添加到对应列表中
        input_texts.append(eng)
        target_texts.append(target)

print(f"Loaded {len(input_texts)} sentence pairs.")

# 显示一些样本数据
print("Sample input texts:", input_texts[:3])
print("Sample target texts:", target_texts[:3])

# ------------------------------
# 2. 文本分词和序列化
# ------------------------------
# 创建英文分词器，保留特殊符号（filters=''）以便不丢失<sos>/<eos>标记
eng_tokenizer = Tokenizer(filters='', oov_token="<OOV>")
# 根据英文文本训练分词器
eng_tokenizer.fit_on_texts(input_texts)
# 将英文文本转换为序列（数字表示）
eng_sequences = eng_tokenizer.texts_to_sequences(input_texts)
# 计算英文句子的最大长度
max_eng_len = max(len(s) for s in eng_sequences)

# 创建中文分词器
cmn_tokenizer = Tokenizer(filters='', oov_token="<OOV>")
# 根据中文文本训练分词器
cmn_tokenizer.fit_on_texts(target_texts)
# 将中文文本转换为序列
cmn_sequences = cmn_tokenizer.texts_to_sequences(target_texts)
# 计算中文句子的最大长度
max_cmn_len = max(len(s) for s in cmn_sequences)

# 计算词汇表大小（+1是为了包含填充符0）
num_eng_tokens = len(eng_tokenizer.word_index) + 1  # +1 for padding (0)
num_cmn_tokens = len(cmn_tokenizer.word_index) + 1

print("English tokens:", num_eng_tokens, "Max length:", max_eng_len)
print("Chinese tokens:", num_cmn_tokens, "Max length:", max_cmn_len)

# 对序列进行填充，使它们具有相同的长度
encoder_input_data = pad_sequences(eng_sequences, maxlen=max_eng_len, padding="post")
decoder_input_data = pad_sequences(cmn_sequences, maxlen=max_cmn_len, padding="post")

print("Encoder input shape:", encoder_input_data.shape)
print("Decoder input shape:", decoder_input_data.shape)

# 解码器目标数据是解码器输入向左移动一位的结果（教师强制训练）
decoder_target_data = np.zeros_like(decoder_input_data)
decoder_target_data[:, :-1] = decoder_input_data[:, 1:]
# 最后一列保持为0（填充）——这对于sparse_categorical_crossentropy是可以接受的

# ---- 数据切分：构造一个可复用的训练/验证集（用于训练并保留验证样本作评估）
indices = np.arange(encoder_input_data.shape[0])
np.random.seed(42)
np.random.shuffle(indices)
val_split = 0.1
val_count = int(len(indices) * val_split)
val_indices = indices[:val_count]
train_indices = indices[val_count:]

encoder_input_train = encoder_input_data[train_indices]
decoder_input_train = decoder_input_data[train_indices]
decoder_target_train = decoder_target_data[train_indices]

encoder_input_val = encoder_input_data[val_indices]
decoder_input_val = decoder_input_data[val_indices]
decoder_target_val = decoder_target_data[val_indices]

print(f"Train samples: {encoder_input_train.shape[0]}, Val samples: {encoder_input_val.shape[0]}")

# ------------------------------
# 3. 构建Seq2Seq模型（训练阶段）
# ------------------------------
print("Building model...")

# 编码器部分
# 输入层：接收任意长度的整数序列
encoder_inputs = Input(shape=(None,), name="encoder_inputs")
# 嵌入层：将整数序列转换为密集向量表示
# Use mask_zero=True so the Embedding produces a mask for padded timesteps (token index 0).
# This lets the LSTM and Keras ignore padded positions when propagating masks and when computing loss
encoder_embedding = Embedding(input_dim=num_eng_tokens, output_dim=embedding_dim, mask_zero=True, name="encoder_embedding")
# 应用嵌入层
enc_emb = encoder_embedding(encoder_inputs)
# 添加dropout层
enc_emb = Dropout(dropout_rate)(enc_emb)
# LSTM层：返回状态（隐藏状态和单元状态）
encoder_lstm = LSTM(latent_dim, return_state=True, name="encoder_lstm")
# 获取LSTM输出和最终状态
_, state_h, state_c = encoder_lstm(enc_emb)
# 编码器的最终状态将作为解码器的初始状态
encoder_states = [state_h, state_c]

# 解码器部分（训练阶段）
# 解码器输入层
decoder_inputs = Input(shape=(None,), name="decoder_inputs")
# 解码器嵌入层
# Decoder embedding also uses mask_zero so decoder ignores padding tokens during training
decoder_embedding = Embedding(input_dim=num_cmn_tokens, output_dim=embedding_dim, mask_zero=True, name="decoder_embedding")
# 应用嵌入层
dec_emb = decoder_embedding(decoder_inputs)
# 添加dropout层
dec_emb = Dropout(dropout_rate)(dec_emb)

# 解码器LSTM层：返回序列和状态
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
# LSTM输出和状态，使用编码器的最终状态作为初始状态
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
# 添加dropout层
decoder_outputs = Dropout(dropout_rate)(decoder_outputs)
# 全连接层：将LSTM输出映射到中文字典大小，使用softmax激活函数输出概率分布
decoder_dense = Dense(num_cmn_tokens, activation="softmax", name="decoder_dense")
decoder_outputs = decoder_dense(decoder_outputs)

# 构建完整的训练模型
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# 编译模型：使用Adam优化器和稀疏分类交叉熵损失函数
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# 显示模型结构摘要
model.summary()

# ------------------------------
# 4. 训练模型
# ------------------------------
print("Training model...")

# 解码器目标数据形状：(样本数, 时间步)
# sparse_categorical_crossentropy期望整数标签，但Keras要求3D输入时形状为(samples, timesteps, 1)
decoder_target_data_expanded = np.expand_dims(decoder_target_data, -1)

# Create sample weights to ignore padding positions (where decoder_target_data == 0).
# Keras will use these weights to mask the loss for padded timesteps so the model doesn't learn
# to always predict the padding index (0).
sample_weights = (decoder_target_data != 0).astype('float32')

checkpoint_path = f"seq2seq_best_ns{num_samples}_ld{latent_dim}.keras"
callbacks = [
    ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
]

# 如果已经存在检查点文件则直接加载，避免重复长时间训练
if os.path.exists(checkpoint_path):
    print(f"Found checkpoint {checkpoint_path}, loading weights and skipping training.")
    model.load_weights(checkpoint_path)
else:
    print("No checkpoint found, starting training...")
    history = model.fit(
        [encoder_input_train, decoder_input_train],
        np.expand_dims(decoder_target_train, -1),
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([encoder_input_val, decoder_input_val], np.expand_dims(decoder_target_val, -1)),
        verbose=1,
        sample_weight=(decoder_target_train != 0).astype('float32'),
        callbacks=callbacks
    )

# ------------------------------
# 5. 构建推理模型
# ------------------------------
print("Building inference models...")

# 推理阶段的编码器模型：输入英文句子，输出编码后的状态
encoder_model = Model(encoder_inputs, encoder_states)

# 推理阶段的解码器模型
# 解码器状态输入：前一个时间步的隐藏状态和单元状态
dec_state_input_h = Input(shape=(latent_dim,), name="dec_state_input_h")
dec_state_input_c = Input(shape=(latent_dim,), name="dec_state_input_c")
decoder_states_inputs = [dec_state_input_h, dec_state_input_c]

# 解码器单步输入：每次只处理一个时间步的token
decoder_single_input = Input(shape=(1,), name="decoder_single_input")  # 单个时间步
# 应用嵌入层（复用训练时的权重）
dec_single_emb = decoder_embedding(decoder_single_input)
# 解码器LSTM层（复用训练时的权重）
dec_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(dec_single_emb, initial_state=decoder_states_inputs)
# 应用全连接层（复用训练时的权重）
dec_outputs_inf = decoder_dense(dec_outputs_inf)  # 输出形状: (batch, 1, num_cmn_tokens)

# 构建推理阶段的解码器模型
decoder_model = Model(
    [decoder_single_input] + decoder_states_inputs,
    [dec_outputs_inf, state_h_inf, state_c_inf]
)

# ------------------------------
# 6. 建立反向索引（id -> word）
# ------------------------------
# 创建从id到单词的映射字典
reverse_cmn_index = getattr(cmn_tokenizer, 'index_word', None) or {idx: word for word, idx in cmn_tokenizer.word_index.items()}
# 确保索引0存在并映射为空字符串（填充符）
reverse_cmn_index[0] = ''

# 安全获取token id的辅助函数
def get_token_id(token):
    return cmn_tokenizer.word_index.get(token, None)

# 获取开始和结束标记的id
sos_id = get_token_id(sos_token)
eos_id = get_token_id(eos_token)
# 检查标记是否正确插入
if sos_id is None or eos_id is None:
    raise ValueError("Start or end token not found in Chinese tokenizer. Check token insertion.")

print("SOS token ID:", sos_id)
print("EOS token ID:", eos_id)

# ------------------------------
# 7. 翻译函数（推理过程）
# ------------------------------
def translate_sentence(input_text, max_len=max_cmn_len):
    """
    将英文句子翻译为中文句子
    
    参数:
    input_text: 待翻译的英文句子
    max_len: 生成中文句子的最大长度
    
    返回:
    翻译后的中文句子
    """
    print(f"Translating: '{input_text}'")
    
    # 预处理输入文本
    seq = eng_tokenizer.texts_to_sequences([input_text.lower()])
    print(f"Input sequence: {seq}")
    
    # 填充序列至固定长度
    seq = pad_sequences(seq, maxlen=max_eng_len, padding="post")
    
    # 检查序列是否为空
    if len(seq[0]) == 0:
        return "无法处理输入文本"

    # 编码输入句子以获取初始状态
    states_value = encoder_model.predict(seq, verbose=0)
    print(f"Encoder states shape: {[s.shape for s in states_value]}")

    # 以开始标记作为解码器的第一个输入（形状为(1,1)）
    target_seq = np.array([[sos_id]])
    
    # 检查sos_id是否有效
    if sos_id is None:
        return "无法找到开始标记"

    # 存储解码得到的单词
    decoded_tokens = []
    # 迭代生成最多max_len个单词
    for i in range(max_len):
        # 解码器预测下一个单词的概率分布
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        # 选择概率最高的单词索引
        probs = output_tokens[0, -1, :]
        sampled_token_index = int(np.argmax(probs))
        # 打印前5个候选token及其概率（用于调试）
        top_k = 5
        top_indices = probs.argsort()[-top_k:][::-1]
        top_candidates = [(int(idx), reverse_cmn_index.get(int(idx), '<UNK>'), float(probs[int(idx)])) for idx in top_indices]
        print(f"Step {i}: top-{top_k} candidates: {top_candidates}")

        # 如果遇到填充符或结束标记则停止
        if sampled_token_index == 0 or sampled_token_index == eos_id:
            print(f"Stop token encountered: {sampled_token_index}")
            break

        # 获取对应索引的单词
        sampled_word = reverse_cmn_index.get(sampled_token_index, "")
        print(f"Sampled word: '{sampled_word}'")
        # 如果找不到对应单词也停止
        if sampled_word == "":
            print("Empty word encountered")
            break

        # 添加到解码结果中
        decoded_tokens.append(sampled_word)

        # 更新解码器输入（之前采样的单词）和状态
        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]

    # 将单词列表连接成句子
    decoded_sentence = " ".join(decoded_tokens)
    result = decoded_sentence.strip() if decoded_sentence else "无翻译结果"
    print(f"Final translation: '{result}'")
    return result

# ------------------------------
# 8. 测试翻译功能
# ------------------------------
print("\n=== Translation Test ===")
# 测试用例
examples = [
    "hi",
    "run",
    "i know"
]

# 对每个例子进行翻译并打印结果
for s in examples:
    print("English:", s)
    print("Chinese (pred):", translate_sentence(s))
    print("---")

# ------------------------------
# Beam search decoder for inference
def beam_search_translate(input_text, beam_width=3, max_len=max_cmn_len):
    # prepare input
    seq = eng_tokenizer.texts_to_sequences([input_text.lower()])
    seq = pad_sequences(seq, maxlen=max_eng_len, padding='post')
    states_value = encoder_model.predict(seq, verbose=0)

    # Beam entries: (token_seq, logprob, states)
    beams = [([sos_id], 0.0, states_value)]

    for _ in range(max_len):
        all_candidates = []
        for token_seq, logprob, states in beams:
            last_token = np.array([[token_seq[-1]]])
            output_tokens, h, c = decoder_model.predict([last_token] + states, verbose=0)
            probs = output_tokens[0, -1, :]
            # take top beam_width candidates
            top_indices = probs.argsort()[-beam_width:][::-1]
            for idx in top_indices:
                p = probs[int(idx)]
                if p <= 0:
                    continue
                candidate_seq = token_seq + [int(idx)]
                candidate_logprob = logprob + np.log(p + 1e-9)
                candidate_states = [h, c]
                all_candidates.append((candidate_seq, candidate_logprob, candidate_states))
        # select best beam_width
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        beams = ordered[:beam_width]
        # if any beam ends with eos, we can stop early (but keep others)
        if any(b[0][-1] == eos_id for b in beams):
            break

    # choose best final beam that ends not with padding; prefer eos-ending
    beams_sorted = sorted(beams, key=lambda x: x[1], reverse=True)
    best_seq = beams_sorted[0][0]
    # strip sos and tokens after eos
    out_tokens = []
    for tid in best_seq:
        if tid == sos_id:
            continue
        if tid == eos_id:
            break
        if tid == 0:
            break
        out_tokens.append(reverse_cmn_index.get(int(tid), ''))
    return ' '.join([t for t in out_tokens if t]) or '无翻译结果'

# ------------------------------
# Evaluate on a small validation sample: print greedy vs beam
def build_reference_from_decoder_row(row):
    # row is decoder_input row (with <sos> at start). Build human-readable reference up to eos.
    tokens = []
    for tid in row:
        if tid == sos_id:
            continue
        if tid == eos_id or tid == 0:
            break
        tokens.append(reverse_cmn_index.get(int(tid), ''))
    return ' '.join([t for t in tokens if t])

def evaluate_on_val(n=20, beam_width=3):
    n = min(n, encoder_input_val.shape[0])
    exact_matches = 0
    token_matches = 0
    total_tokens = 0
    print(f"\n=== Evaluation on {n} val samples (greedy vs beam={beam_width}) ===")
    for i in range(n):
        src = ' '.join([w for w in eng_tokenizer.sequences_to_texts([encoder_input_val[i]])[0].split() if w])
        ref = build_reference_from_decoder_row(decoder_input_val[i])
        greedy = translate_sentence(src)
        beam = beam_search_translate(src, beam_width=beam_width)
        print(f"SRC: {src}")
        print(f"REF: {ref}")
        print(f"GREEDY: {greedy}")
        print(f"BEAM: {beam}")
        print('---')
        if beam == ref:
            exact_matches += 1
        # token overlap (naive)
        ref_tokens = ref.split()
        beam_tokens = beam.split()
        total_tokens += len(ref_tokens)
        token_matches += sum(1 for t in beam_tokens if t in ref_tokens)
    print(f"Exact match (beam): {exact_matches}/{n}")
    print(f"Token-level match rate: {token_matches}/{total_tokens} = {token_matches/total_tokens if total_tokens>0 else 0:.3f}")

# Run evaluation on validation set (beam vs greedy)
evaluate_on_val(n=20, beam_width=3)
