"""
word2vec
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors
import math
model_path = 'sgns.weibo.word.bz2'
model = KeyedVectors.load_word2vec_format(model_path)
#加载已有的词向量文件
#查看词向量维度
print(model.vector_size)
#查看某个词的向量
print(model['地铁'])
#查看两个向量的相似度
similarity = model.similarity('地铁', '公交')
print('地铁 vs 公交 相似度：', similarity)

from gensim.models import Word2Vec#用来训练自己的词向量的文件

sentences = [['我', '每天','乘坐', '地铁', '上班'], ['我','每天', '乘坐', '公交', '上班']]

model = Word2Vec(
          sentences,        # 已分词的句子序列
          vector_size=100,  # 词向量维度
          window=5,         # 上下文窗口大小
          min_count=2,      # 最小词频（低于将被忽略）
          sg=1,             # 1:Skip-Gram，0:CBOW
          workers=4         # 并行训练线程数
)

#保存词向量
model.wv.save_word2vec_format('my_vectors.kv')
#加载词向量
from gensim.models import KeyedVectors

my_model = KeyedVectors.load_word2vec_format('my_vectors.kv')
"""
加载训练好的词向量（如 Word2Vec）到嵌入层中作为初始参数，这样可以为模型注入丰富的语言知识，
尤其在低资源任务中优势明显。并且，加载预训练词向量后，可选择是否让嵌入层继续参与训练
"""
#由于报错看的难受，把下面的粘贴上来
word_vectors = KeyedVectors.load_word2vec_format("my_vectors.kv")
word2index = word_vectors.key_to_index # 词到索引的映射
embedding_dim = word_vectors.vector_size # 词语向量维度
num_embeddings = len(word2index) # 词表大小

embedding_matrix = torch.zeros(num_embeddings, embedding_dim) # 构造词向量矩阵,形状为(词表大小,词向量维度大小)
#演示如何使用预训练词向量初始化Embedding层
embedding_layer = nn.Embedding.from_pretrained(
          embedding_matrix, # 词向量矩阵，形状为(num_embeddigns,embedding_dim)
          freeze=False  # 是否冻结词向量
)

"""
示例一下怎么把word2vec用在工程的词嵌入层中
"""
# 1. 加载预训练的 Word2Vec 模型
word_vectors = KeyedVectors.load_word2vec_format("my_vectors.kv")

# 2. 构建词表和词向量矩阵
word2index = word_vectors.key_to_index # 词到索引的映射
embedding_dim = word_vectors.vector_size # 词语向量维度
num_embeddings = len(word2index) # 词表大小

embedding_matrix = torch.zeros(num_embeddings, embedding_dim) # 构造词向量矩阵,形状为(词表大小,词向量维度大小)
for word, idx in word2index.items():
         embedding_matrix[idx] = torch.tensor(word_vectors[word])

# 3. 构建 PyTorch 的嵌入层
embedding_layer = nn.Embedding.from_pretrained(
          embedding_matrix, # 词向量矩阵，形状为(num_embeddigns,embedding_dim)
          freeze=False # 是否冻结词向量
)
#这里依旧是，得把词向量转化为索引表

# 4. 示例：将词索引转换为向量
input_words = ["我", "喜欢", "乘坐", "地铁"] # 分词后的句子
input_indices = [word2index[word] for word in input_words] # token转为索引
input_tensor = torch.tensor([input_indices]) # 构造嵌入层输入张量

# 5. 查询嵌入（即词向量查找）
output = embedding_layer(input_tensor) # 通过嵌入层查找预训练词向量

print(output.shape) # 例如 torch.Size([1, 4, 100])
#这里注意输入得转换成张量，torch中一切皆张量

"""
RNN
"""
torch.nn.RNN(
         input_size= 3,
         hidden_size= 32,       #隐藏状态的维度，决定”记忆的容量“
         num_layers=1,
         nonlinearity="tanh",
         bias=True,
         batch_first=False,
         dropout=0.0,
         bidirectional=False,
         device=None,
         dtype=None,
)
"""
一个NLP任务的一般步骤，一般分多个py文件实现：
1，数据预处理：读取数据集，清洗数据，划分数据集，tokenizer,保存，这里需要熟练掌握pandas库的用法
2，tokenizer:相当于提前写好一个“工具”，方便复用
3，自定义数据集：自己写个数据集的类，继承至pytorch的dataset父类，将json文件转换为按批次的torch张量
4，模型定义：自己写神经网络结构的类，包括词嵌入层这些，有的encoder,decoder得分开来写
5，模型训练脚本：一般先写一个epoch运转的函数，再写一个总的训练的函数
6，模型预测脚本：如果模型有个什么预测功能的话，用于实现模型的效果检验
7，模型评估：评价一下这个模型，准确率之类的，可以结合可视化
8，配置文件：将一些需要的超参以常量形式统一写在config。py文件中

"""
"""
训练脚本的基本逻辑，只是基本，按需改进
"""
def train_one_epoch(model, data_loader,loss_function,optimizer, device):
    """
       训练一个 epoch。

       :param model: 输入法模型。
       :param dataloader: 数据加载器。
       :param loss_function: 损失函数。
       :param optimizer: 优化器。
       :param device: 设备。
       :return: 平均损失。
       """

         total_loss = 0
         model.train()

        for inputs, targets in tqdm(dataloader, desc='训练'):
        # 将数据移到设备
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

        # 前向传播
            outputs = model(inputs)

        # 计算损失
            loss = loss_function(outputs, targets)

        # 反向传播
            loss.backward()

        # 更新参数
            optimizer.step()

            total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            return avg_loss


    def train():
                """
                模型训练主函数。
                """

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print('设备:', device)

     # 获取数据加载器
            dataloader = get_dataloader()

              # 加载 tokenizer 和模型
            tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')
            model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)

            loss_function = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

      # TensorBoard 日志
            writer = SummaryWriter(log_dir=config.LOG_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

            best_loss = float('inf')

            for epoch in range(1, config.EPOCHS + 1):
                print(f'========== Epoch: {epoch} ==========')

                    # 训练一个 epoch
                avg_loss = train_one_epoch(model, dataloader, loss_function, optimizer, device)
                print(f'Loss: {avg_loss:.4f}')

            # 记录到 TensorBoard
                writer.add_scalar('Loss/train', avg_loss, epoch)

            # 保存最优模型
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(model.state_dict(), config.MODELS_DIR / 'model.pt')
                    print('模型保存成功！')

    if __name__ == '__main__':
            train()

"""
手搓一下RNN,LSTM,GRU的基本类，有助于更好的理解RNN
"""
class myRNN(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(myRNN, self).__init__()
        self.input_size = input_size    #输入维度
        self.hidden_size = hidden_size  #隐藏层维度
        self.linear = nn.Linear(input_size+ hidden_size,hidden_size)
    def forward(self, x,h_prev):
        concat = torch.cat([x,h_prev],1)
        h_t = torch.tanh(self.linear(concat))

        return h_t
class myLSTM(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(myLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        #遗忘门
        self.w_f = nn.Linear(input_size+hidden_size,hidden_size)
        #输入门
        self.w_i = nn.Linear(input_size+hidden_size,hidden_size)
        #细胞状态
        self.w_c = nn.Linear(input_size+hidden_size,hidden_size)
        #输出门
        self.w_o = nn.Linear(input_size+hidden_size,hidden_size)
    def forward(self, x,h_prev,c_prev):
        concat = torch.cat([x,h_prev],1)
        #遗忘门
        f_t = torch.sigmoid(self.w_f(concat))
        #输入门+候选细胞
        i_t = torch.sigmoid(self.w_i(concat))
        c_ti = torch.tanh(self.w_c(concat))
        #更新细胞状态
        c_t = i_t * c_ti + f_t * c_prev
        #输出门
        o_t = torch.sigmoid(self.w_o(concat))
        h_t = o_t * c_t
        return h_t,c_t

class myGRU(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(myGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        #更新门
        self.w_z = nn.Linear(input_size+hidden_size,hidden_size)
        #重置门
        self.w_r = nn.Linear(input_size+hidden_size,hidden_size)
        #候选隐藏状态
        self.w_h = nn.Linear(input_size+hidden_size,hidden_size)

    def forward(self, x,h_prev):
        concat = torch.cat([x,h_prev],1)
        #更新门+重置门
        r_t = torch.sigmoid(self.w_z(concat))
        z_t = torch.sigmoid(self.w_r(concat))
        #先重置旧记忆，再拼接
        concat_reset = torch.cat([x,r_t*h_prev],1)
        h_tilde = torch.tanh(self.w_h(concat_reset))
        #隐藏状态更新
        h_t = (1-z_t) * h_prev + z_t * h_tilde
        return h_t
"""
手写一下transformer的类
"""
def create_padding_mask(seq,pad_idx):
    #将pad的部分忽略掉因为pad只是用来占位
    mask = (seq == pad_idx).unsqueeze(1).unsqueeze(2)
    return mask

def create_subsequent_mask(seq_len):
    #掩码自注意力机制，防止解码器看到未来的消息
    mask = torch.triu(torch.ones(1,1,seq_len,seq_len), diagonal=1)
    #1的部分为true，0的部分为false
    return mask == 0

#缩放点积注意力
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):

        d_k = Q.size(-1)
        #q乘k的转置再除以根号d_k
        scores = torch.matmul(Q, K.transpose(-1,-2)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attn_weights, V)

        return output, attn_weights

#多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_heads,dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        #映射Q,K,V
        self.W_q = nn.Linear(d_model, n_heads)
        self.W_k = nn.Linear(d_model, n_heads)
        self.W_v = nn.Linear(d_model, n_heads)

        #输出的线性层
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model,eps=1e-6)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        #保存残差连接
        residual = Q

        #线性变换并分头
        Q = self.W_q(Q).view(batch_size,-1,self.n_heads,self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size,-1,self.n_heads,self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size,-1,self.n_heads,self.d_k).transpose(1, 2)

        attn_output, attention_weights = self.attention(Q, K, V, mask)

        atttn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)

        output = self.W_o(atttn_output)

        output = self.layer_norm(self.dropout(output) + residual)

        return output, attention_weights

#前馈神经网络
class PositionWiseFeedEncoding(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedEncoding, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.lay_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        residual = x

        output = self.lay_norm(residual+self.fc2(self.dropout(torch.relu(self.fc1(x)))))

        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        #初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model)

        #生成位置向量
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        #计算分母项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        #偶数位置用sin,奇数用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        #增加维度
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :].detach()
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads,d_ff,dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.ffn = PositionWiseFeedEncoding(d_model, d_ff, dropout=dropout)

        def forword(self,x,mask):
            #自注意力
            attn_output,attention_weights = self.self_attn(x,x,x,mask)

            #前馈网络
            ffn_output = self.ffn(attn_output)

            return ffn_output,attention_weights
class Encoder(nn.Module):
    def __init__(self, src_vocab_size,n_layers, d_model, n_heads, d_ff, max_len,dropout,pad_idx):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, src, src_mask):
        #词嵌入加位置编码

        x = self.src_embedding(src)*math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        attn_weights_list = []
        for layer in self.layers:
            x,attn_weights = layer(x,src_mask)
            attn_weights_list.append(attn_weights)

        return x, attn_weights_list
class Decoderlayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(Decoderlayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.ffn = PositionWiseFeedEncoding(d_model, d_ff, dropout=dropout)

    def forward(self, x, enc_output, src_mask,tgt_mask):
        #掩码多头注意力
        dec_output, dec_attn_weights = self.self_attn(x, x, x, tgt_mask)

        #交叉注意力
        cross_output,cross_attn_weights = self.cross_attn(dec_output, enc_output, enc_output, src_mask)

        #前馈神经网络
        ffn_output = self.ffn(cross_output)

        return ffn_output, dec_attn_weights,cross_attn_weights
class Decoder(nn.Module):
    def __init__(self,tgt_vocab_size,n_layers,d_model,n_heads,d_ff,dropout,max_len,pad_idx):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        self.layers = nn.ModuleList([Decoderlayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, tgt, enc_output, src_mask,tgt_mask):
        #词嵌入加位置编码
        x = self.tgt_embedding(tgt)*math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        dec_attn_weights_list = []
        cross_attn_weights_list = []

        for layer in self.layers:
            x,dec_attn,cross_attn = layer(x,enc_output,src_mask,tgt_mask)
            dec_attn_weights_list.append(dec_attn)
            cross_attn_weights_list.append(cross_attn)

        return x, dec_attn_weights_list,cross_attn_weights_list

class transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx,
                 d_model=512, n_layers=6, n_heads=8, d_ff=2048, max_len=5000, dropout=0.1):
        super(transformer, self).__init__()

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.encoder = Encoder(src_vocab_size, n_layers, d_model, n_heads, d_ff, max_len, dropout,src_pad_idx)
        self.decoder = Decoder(tgt_vocab_size, n_layers, d_model, n_heads, d_ff, max_len, dropout,tgt_pad_idx)

        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def make_src_mask(self, src):
        return create_padding_mask(src,self.src_pad_idx)

    def make_tgt_mask(self, tgt):
        pad_mask = create_padding_mask(tgt,self.tgt_pad_idx)
        subsequent_mask = create_padding_mask(tgt.size(1)).to(tgt.device)
        return pad_mask & subsequent_mask

    def forward(self, src, tgt, src_mask, tgt_mask):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        #编码器向前传播
        enc_output,enc_att = self.encoder(src, src_mask)

        #解码器向前传播
        dec_output,dec_attns,cross_attns = self.decoder(tgt, enc_output, src_mask,tgt_mask)

        #最后线性映射
        output = self.fc(dec_output)

        return output,enc_att,dec_attns,cross_attns
def main():





















