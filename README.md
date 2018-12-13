Keras——文本与序列预处理
===================================

# 一. 简介

> 在进行自然语言处理之前，需要对文本进行处理。 本文介绍keras提供的预处理包keras.preproceing下的text与序列处理模块sequence模块。

# 二. text提供的方法

> text提供了分词和one-hot编码的方法用于文本处理，函数功能如下：

## 1. text_to_word_sequence

```python
keras.preprocessing.text.text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[]^_`{|}~\t\n', lower=True, split=" ")
```

> 本函数将一个句子拆分成单词构成的列表，功能相当于str.split，参数解释如下：
>
> - text：字符串，待处理的文本
> - filters：需要滤除的字符的列表或连接形成的字符串，例如标点符号。默认值为 '!"#$%&()*+,-./:;<=>?@[]^_`{|}~\t\n'，包含标点符号，制表符和换行符等
> - lower：布尔值，是否将序列设为小写形式
> - split：字符串，单词的分隔符，如空格

```python
import keras.preprocessing.text as T
text = 'some thing to eat'
print(T.text_to_word_sequence(text))


结果如下：
['some', 'thing', 'to', 'eat']
```

## 2. one_hot

```python
keras.preprocessing.text.one_hot(text, n, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n', lower=True, split=" ")
```

> 本函数将一段文本编码为one-hot形式的码，即仅记录词在词典中的下标。
>
> 从定义上，当字典长为n时，每个单词应形成一个长为n的向量，其中仅有单词本身在字典中下标的位置为1，其余均为0，这称为one-hot。为了方便起见，函数在这里仅把“1”的位置，即字典中词的下标记录下来。
>
> - n：整数，字典长度

```python
import keras.preprocessing.text as T
text = 'some thing to eat'
print(T.one_hot(text, 10))

结果如下：
[6, 2, 8, 4]
多次运行结果会有所不同
```

# 三. text.Tokenizer

```python
keras.preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n', lower=True, split=" ", char_level=False)
```

> Tokenizer是一个用于向量化文本，或将文本转换为序列（即单词在字典中的下标构成的列表，从1算起）的类。
>
> - 与`text_to_word_sequence`同名参数含义相同
> - num_words：None或整数，处理的最大单词数量。若被设置为整数，则分词器将被限制为待处理数据集中最常见的`num_words`个单词
> - char_level: 如果为 True, 每个字符将被视为一个标记

## (一) 类方法

### 1. fit_on_texts(texts)

> 使用一系列文档来生成token词典，texts为list类，每个元素为一个文档。

### 2. texts_to_sequences(texts)

> 将多个文档转换为word下标的向量形式。
>
> texts：待转为序列的文本列表
>
> 返回值：序列的列表，列表中每个序列对应于一段输入文本，shape为[文档数，每条文档的长度]

### 3. texts_to_matrix(texts, mode)

> 将多个文档转换为矩阵表示，shape为[文档数，词数]
>
> texts：待向量化的文本列表
>
> mode：‘binary’，‘count’，‘tfidf’，‘freq’之一，默认为‘binary’
>
> 返回值：形如`(len(texts), nb_words)`的numpy array

```python
import keras.preprocessing.text as T
from keras.preprocessing.text import Tokenizer

text1 = 'some thing to eat'
text2 = 'some thing to drink'
texts = [text1, text2]

print(T.text_to_word_sequence(text1))
print(T.one_hot(text1, 10))

tokenizer = Tokenizer(num_words=None)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts) #受num_words影响
print(sequences)

sequences_matrix = tokenizer.texts_to_matrix(texts)
print(sequences_matrix)

结果：
['some', 'thing', 'to', 'eat']

[9, 3, 7, 3]

[[1, 2, 3, 4], [1, 2, 3, 5]]

[[0. 1. 1. 1. 1. 0.]
 [0. 1. 1. 1. 0. 1.]]
```

## (二) 属性

### 1. word_counts

>  字典，将单词（字符串）映射为它们在训练期间所有文档中出现的次数，仅在调用fit_on_texts之后设置。

### 2. word_docs

> 字典，将单词（字符串）映射为它们在训练期间所出现的文档或文本的数量，仅在调用fit_on_texts之后设置。

### 3. word_index

> 字典，将单词（字符串）映射为它们的排名或者索引id，仅在调用fit_on_texts之后设置。

### 4. index_docs

> 字典，保存word的id出现的文档的数量

```python
import keras.preprocessing.text as T
from keras.preprocessing.text import Tokenizer

text1 = 'some thing to eat'
text2 = 'some thing to drink'
texts = [text1, text2]

print(T.text_to_word_sequence(text1))
print(T.one_hot(text1, 10))

tokenizer = Tokenizer(num_words=None)
tokenizer.fit_on_texts(texts)

print(tokenizer.word_counts)

print(tokenizer.word_index)

print(tokenizer.word_docs)

print(tokenizer.index_docs)

结果：
OrderedDict([('some', 2), ('thing', 2), ('to', 2), ('eat', 1), ('drink', 1)])

{'some': 1, 'thing': 2, 'to': 3, 'eat': 4, 'drink': 5}

defaultdict(<class 'int'>, {'thing': 2, 'eat': 1, 'to': 2, 'some': 2, 'drink': 1})

defaultdict(<class 'int'>, {2: 2, 4: 1, 3: 2, 1: 2, 5: 1})
```

# 四. sequence

##  (一) 填充序列pad_sequences

```python
keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.)
```

> 将长为`nb_samples`的序列（标量序列）转化为形如`(nb_samples,nb_timesteps)`2D numpy array。如果提供了参数`maxlen`，`nb_timesteps=maxlen`，否则其值为最长序列的长度。其他短于该长度的序列都会在后部填充0以达到该长度。长于`nb_timesteps`的序列将会被截断，以使其匹配目标长度。padding和截断发生的位置分别取决于`padding`和`truncating`.
>
> - sequences：浮点数或整数构成的两层嵌套列表
>
> - maxlen：None或整数，为序列的最大长度。大于此长度的序列将被截短，小于此长度的序列将在后部填0.
>
> - dtype：返回的numpy array的数据类型
>
> - padding：‘pre’或‘post’，确定当需要补0时，在序列的起始还是结尾补
>
> - truncating：‘pre’或‘post’，确定当需要截断序列时，从起始还是结尾截断
>
> - value：浮点数，此值将在填充时代替默认的填充值0
>
> 返回形如`(nb_samples,nb_timesteps)`的2D张量

```python
from keras.preprocessing import sequence
import keras.preprocessing.text as T
from keras.preprocessing.text import Tokenizer

text1 = 'some thing to eat'
text2 = 'some thing to drink'
texts = [text1, text2]

tokenizer = Tokenizer(num_words=None)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts) #受num_words影响
print(sequence.pad_sequences(sequences, maxlen=10))

结果：
[[0 0 0 0 0 0 1 2 3 4]
 [0 0 0 0 0 0 1 2 3 5]]
```

参考：
https://keras-cn.readthedocs.io/en/latest/preprocessing/sequence/.

https://blog.csdn.net/lovebyz/article/details/77712003.
