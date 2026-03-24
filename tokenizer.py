from abc import abstractmethod                                 #抽象方法类
from nltk import word_tokenize, TreebankWordTokenizer          #英语分词器，逆分词工具、
from tqdm import tqdm                                          #进度条工具
import jieba                                                   #中文分词工具

jieba.setLogLevel(jieba.logging.WARNING)
#通用tokenizer父类
class Basetokenizer():

    #特殊标记，保证格式一致
    unk_token = '<UNK>'
    pad_token = '<pad>'
    sos_token = '<sos>'
    eos_token = '</sos>'

    @staticmethod
    @abstractmethod
    def tokenize(sentence):
       pass

    @abstractmethod
    def decode(self,sentence):
       pass


    @classmethod
    def build_vocab(cls, sentences,vocab_file):
       #将每个词收集在这个集合里，去重
        unique_words = set()
        for sentence in tqdm(sentences,desc="分词"):
            for word in cls.tokenize(sentence):
                unique_words.add(word)
        vocab_list = [cls.pad_token+cls.unk_token+cls.sos_token+cls.eos_token] + list(unique_words)
        #将构建的词表保存在文件里
        with open(vocab_file,'w',encoding='utf-8') as f:
            for word in vocab_list:
                f.write(word+'\n')

    @classmethod
    #读取文件里的词表
    def from_vocab(cls, vocab_file):
        with open(vocab_file,'r',encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f.readlines()]
        return cls(vocab_list)
    def __init__(self,vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        #词到索引映射
        self.word2index = {word: index for index, word in enumerate(self.vocab_list)}
        #索引到词的映射
        self.index2word = {index:word for index, word in enumerate(self.vocab_list)}
        #获取未知词的索引
        self.unk_token_index = self.word2index[self.unk_token]
        self.sos_token_index = self.word2index[self.sos_token]
        self.eos_token_index = self.word2index[self.eos_token]
        self.pad_token_index = self.word2index[self.pad_token]

    def encode(self,sentence,seq_len,add_sos_eos = False):
        tokens = self.tokenize(sentence)
        indexes = [self.word2index.get(token,self.unk_token_index) for token in tokens]
        #选择是否添加前后
        if add_sos_eos:
            indexes = indexes[:seq_len-2]
            indexes = [self.sos_token_index] + indexes + [self.eos_token_index]
        else:
            indexes = indexes[:seq_len]

        #短句子补齐
        if len(indexes) < seq_len:
            indexes += ([self.pad_token_index] * (seq_len-len(indexes)))

        return indexes
class ChineseTokenizer(Basetokenizer):
    @staticmethod
    def tokenize(sentence):
        return list(jieba.cut(sentence))
    def decode(self,indexes):
        return ''.join([self.index2word[index] for index in indexes])
class EnglishTokenizer(Basetokenizer):
    @staticmethod
    def tokenize(sentence):
        return word_tokenize(sentence)
    def decode(self,indexes):
        tokens = [self.index2word[index] for index in indexes]
        return TreebankWordTokenizer().detokenize(tokens)
"""
使用指南：
# 构建中文词表
ChineseTokenizer.build_vocab(
    sentences=["我 爱 中国", "今天 天气 好"], 
    vocab_file="vocab_chinese.txt"
)

# 构建英文词表
EnglishTokenizer.build_vocab(
    sentences=["I love China", "This is good"], 
    vocab_file="vocab_english.txt"
)

# 加载中文分词器
ch_tokenizer = ChineseTokenizer.from_vocab("vocab_chinese.txt")

# 加载英文分词器
en_tokenizer = EnglishTokenizer.from_vocab("vocab_english.txt")
# 中文编码（长度=10，加开头结尾标记）
ids = ch_tokenizer.encode(
    sentence="我喜欢机器学习",
    seq_len=10,
    add_sos_eos=True
)
print(ids)  # 输出：数字列表

sentence = ch_tokenizer.decode(ids)
print(sentence)  # 输出：我喜欢机器学习

print(ch_tokenizer.vocab_size)
"""













