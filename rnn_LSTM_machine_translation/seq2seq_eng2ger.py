# seq2seq_lstm_en_de_fixed.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Step 0: download data from https://www.manythings.org/anki/
# ------------------------------

# ------------------------------
# 配置参数
# ------------------------------
# 数据路径：manythings.org提供的德语数据集 (格式为: 英文 \t 德文 \t ...)
data_path = "deu.txt"
# 使用的样本数量，用于演示的小子集
num_samples = 10000
# LSTM隐藏层维度
latent_dim = 32
# 词嵌入维度
embedding_dim = 128
# 批次大小
batch_size = 64
# 训练轮数
epochs = 8

# ------------------------------
# 1. 加载和预处理数据
# ------------------------------
# 存储英文句子和德文句子的列表
input_texts = []
target_texts = []

# 检查数据文件是否存在
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}. Download and place 'deu.txt' here.")

# 定义开始和结束标记
sos_token = "<sos>"  # Start of Sentence 标记
eos_token = "<eos>"  # End of Sentence 标记

# 逐行读取数据文件
with open(data_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        # 只读取指定数量的样本
        if i >= num_samples:
            break
        # 分割英文和德文句子（通过制表符）
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        eng, ger = parts[0], parts[1]
        # 转换为小写（可选）
        eng = eng.lower()
        ger = ger.lower()

        # 在德文句子前后添加开始和结束标记
        target = sos_token + " " + ger + " " + eos_token

        # 添加到对应列表中
        input_texts.append(eng)
        target_texts.append(target)

print(f"Loaded {len(input_texts)} sentence pairs.")

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

# 创建德文分词器
ger_tokenizer = Tokenizer(filters='', oov_token="<OOV>")
# 根据德文文本训练分词器
ger_tokenizer.fit_on_texts(target_texts)
# 将德文文本转换为序列
ger_sequences = ger_tokenizer.texts_to_sequences(target_texts)
# 计算德文句子的最大长度
max_ger_len = max(len(s) for s in ger_sequences)

# 计算词汇表大小（+1是为了包含填充符0）
num_eng_tokens = len(eng_tokenizer.word_index) + 1  # +1 for padding (0)
num_ger_tokens = len(ger_tokenizer.word_index) + 1

print("English tokens:", num_eng_tokens, "Max length:", max_eng_len)
print("German tokens:", num_ger_tokens, "Max length:", max_ger_len)

# 对序列进行填充，使它们具有相同的长度
encoder_input_data = pad_sequences(eng_sequences, maxlen=max_eng_len, padding="post")
decoder_input_data = pad_sequences(ger_sequences, maxlen=max_ger_len, padding="post")

# 解码器目标数据是解码器输入向左移动一位的结果（教师强制训练）
decoder_target_data = np.zeros_like(decoder_input_data)
decoder_target_data[:, :-1] = decoder_input_data[:, 1:]
# 最后一列保持为0（填充）——这对于sparse_categorical_crossentropy是可以接受的

# ------------------------------
# 3. 构建Seq2Seq模型（训练阶段）
# ------------------------------
# 编码器部分
# Encoder Layer 1 (Input)：接收任意长度的整数序列
encoder_inputs = Input(shape=(None,), name="encoder_inputs")
# Encoder Layer 2 (Embedding)：将整数序列转换为密集向量表示
enc_emb = Embedding(input_dim=num_eng_tokens, output_dim=embedding_dim, name="encoder_embedding")(encoder_inputs)
# Encoder Layer 3 (LSTM)：返回状态（隐藏状态和单元状态）
_, state_h, state_c = LSTM(latent_dim, return_state=True, name="encoder_lstm")(enc_emb)
# 编码器的最终状态将作为解码器的初始状态
encoder_states = [state_h, state_c]

# 解码器部分（训练阶段）
# Decoder Layer 1 (Input): 解码器输入层
decoder_inputs = Input(shape=(None,), name="decoder_inputs")
# Decoder Layer 2 (Embedding): 解码器嵌入层
dec_embedding_layer = Embedding(input_dim=num_ger_tokens, output_dim=embedding_dim, name="decoder_embedding")
dec_emb = dec_embedding_layer(decoder_inputs)

# Decoder Layer 3 (LSTM)：返回序列和状态
dec_lstm_layer = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
decoder_outputs, _, _ = dec_lstm_layer(dec_emb, initial_state=encoder_states)

# Decoder Layer 4 (Dense)：将LSTM输出映射到德文字典大小，使用softmax激活函数输出概率分布
dec_dense_layer = Dense(num_ger_tokens, activation="softmax", name="decoder_dense")
decoder_outputs = dec_dense_layer(decoder_outputs)

# 构建完整的训练模型
model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name="seq2seq_model_train")
# 编译模型：使用Adam优化器和稀疏分类交叉熵损失函数
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# 显示模型结构摘要
model.summary()

# ------------------------------
# 5. 构建推理模型
# ------------------------------
# 推理阶段的编码器模型：输入英文句子，输出编码后的状态
encoder_model = Model(encoder_inputs, encoder_states, name="encoder_model_inference")
encoder_model.summary()

# 推理阶段的解码器模型
# 解码器状态输入：前一个时间步的隐藏状态和单元状态
decoder_states_inputs = [Input(shape=(latent_dim,), name="dec_state_input_h"),
                         Input(shape=(latent_dim,), name="dec_state_input_c")]

# 解码器单步输入：每次只处理一个时间步的token
decoder_single_input = Input(shape=(1,), name="decoder_single_input")  # 单个时间步
# 应用嵌入层（复用训练时的权重）
dec_single_emb = dec_embedding_layer(decoder_single_input)
# 解码器LSTM层（复用训练时的权重）
dec_outputs_inf, state_h_inf, state_c_inf = dec_lstm_layer(dec_single_emb, initial_state=decoder_states_inputs)
# 应用全连接层（复用训练时的权重）
dec_outputs_inf = dec_dense_layer(dec_outputs_inf)  # 输出形状: (batch, 1, num_ger_tokens)

# 构建推理阶段的解码器模型
decoder_model = Model(
    [decoder_single_input] + decoder_states_inputs,
    [dec_outputs_inf, state_h_inf, state_c_inf],
    name="decoder_model_inference"
)
decoder_model.summary()


# ------------------------------
# 4. 训练模型
# ------------------------------
# 解码器目标数据形状：(样本数, 时间步)
# sparse_categorical_cross_entropy期望整数标签，但Keras要求3D输入时形状为(samples, time_steps, 1)
decoder_target_data_expanded = np.expand_dims(decoder_target_data, -1)

# 开始训练模型
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data_expanded,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.15  # 使用15%的数据作为验证集
)

# ------------------------------
# 6. 建立反向索引（id -> word）
# ------------------------------
# 创建从id到单词的映射字典
reverse_ger_index = {idx: word for word, idx in ger_tokenizer.word_index.items()}
# 注意：填充符0映射为空字符串
reverse_ger_index[0] = ''


# 安全获取token id的辅助函数
def get_token_id(token):
    return ger_tokenizer.word_index.get(token, None)


# 获取开始和结束标记的id
sos_id = get_token_id(sos_token)
eos_id = get_token_id(eos_token)
# 检查标记是否正确插入
if sos_id is None or eos_id is None:
    raise ValueError("Start or end token not found in German tokenizer. Check token insertion.")


# ------------------------------
# 7. 翻译函数（推理过程）
# ------------------------------
def translate_sentence(input_text, max_len=max_ger_len):
    """
    将英文句子翻译为德文句子
    
    参数:
    input_text: 待翻译的英文句子
    max_len: 生成德文句子的最大长度
    
    返回:
    翻译后的德文句子
    """
    # 预处理输入文本
    seq = eng_tokenizer.texts_to_sequences([input_text.lower()])
    # 填充序列至固定长度
    seq = pad_sequences(seq, maxlen=max_eng_len, padding="post")

    # 编码输入句子以获取初始状态
    states_value = encoder_model.predict(seq, verbose=0)

    # 以开始标记作为解码器的第一个输入（形状为(1,1)）
    target_seq = np.array([[sos_id]])

    # 存储解码得到的单词
    decoded_tokens = []
    # 迭代生成最多max_len个单词
    for _ in range(max_len):
        # 解码器预测下一个单词的概率分布
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        # 选择概率最高的单词索引
        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        # 如果遇到填充符或结束标记则停止
        if sampled_token_index == 0 or sampled_token_index == eos_id:
            break

        # 获取对应索引的单词
        sampled_word = reverse_ger_index.get(sampled_token_index, "")
        if sampled_word == "":
            break

        # 添加到解码结果中
        decoded_tokens.append(sampled_word)

        # 更新解码器输入（之前采样的单词）和状态
        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]

    # 将单词列表连接成句子
    decoded_sentence = " ".join(decoded_tokens)
    return decoded_sentence.strip()


# ------------------------------
# 8. 测试翻译功能
# ------------------------------
print("\n=== Translation Test ===")
# 测试用例
examples = [
    "how are you?",
    "i love you",
    "what is your name?",
    "where is the bank?"
]

# 对每个例子进行翻译并打印结果
for s in examples:
    print("English:", s)
    print("German (pred):", translate_sentence(s))
    print("---")
