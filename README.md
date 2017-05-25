
![](https://i.imgur.com/eBRPvWB.png)

# Practical PyTorch: Translation with a Sequence to Sequence Network and Attention

In this project we will be teaching a neural network to translate from French to English.

```
[KEY: > input, = target, < output]

> il est en train de peindre un tableau .
= he is painting a picture .
< he is painting a picture .

> pourquoi ne pas essayer ce vin delicieux ?
= why not try that delicious wine ?
< why not try that delicious wine ?

> elle n est pas poete mais romanciere .
= she is not a poet but a novelist .
< she not not a poet but a novelist .

> vous etes trop maigre .
= you re too skinny .
< you re all alone .
```

... to varying degrees of success.

This is made possible by the simple but powerful idea of the [sequence to sequence network](http://arxiv.org/abs/1409.3215), in which two recurrent neural networks work together to transform one sequence to another. An encoder network condenses an input sequence into a single vector, and a decoder network unfolds that vector into a new sequence.

To improve upon this model we'll use an [attention mechanism](https://arxiv.org/abs/1409.0473), which lets the decoder learn to focus over a specific range of the input sequence.

# The Sequence to Sequence model

A [Sequence to Sequence network](http://arxiv.org/abs/1409.3215), or seq2seq network, or [Encoder Decoder network](https://arxiv.org/pdf/1406.1078v3.pdf), is a model consisting of two separate RNNs called the **encoder** and **decoder**. The encoder reads an input sequence one item at a time, and outputs a vector at each step. The final output of the encoder is kept as the **context** vector. The decoder uses this context vector to produce a sequence of outputs one step at a time.

![](https://i.imgur.com/tVtHhNp.png)

When using a single RNN, there is a one-to-one relationship between inputs and outputs. We would quickly run into problems with different sequence orders and lengths that are common during translation. Consider the simple sentence "Je ne suis pas le chat noir" &rarr; "I am not the black cat". Many of the words have a pretty direct translation, like "chat" &rarr; "cat". However the differing grammars cause words to be in different orders, e.g. "chat noir" and "black cat". There is also the "ne ... pas" &rarr; "not" construction that makes the two sentences have different lengths.

With the seq2seq model, by encoding many inputs into one vector, and decoding from one vector into many outputs, we are freed from the constraints of sequence order and length. The encoded sequence is represented by a single vector, a single point in some N dimensional space of sequences. In an ideal case, this point can be considered the "meaning" of the sequence.

This idea can be extended beyond sequences. Image captioning tasks take an [image as input, and output a description](https://arxiv.org/abs/1411.4555) of the image (img2seq). Some image generation tasks take a [description as input and output a generated image](https://arxiv.org/abs/1511.02793) (seq2img). These models can be referred to more generally as "encoder decoder" networks.

## The Attention Mechanism

The fixed-length vector carries the burden of encoding the the entire "meaning" of the input sequence, no matter how long that may be. With all the variance in language, this is a very hard problem. Imagine two nearly identical sentences, twenty words long, with only one word different. Both the encoders and decoders must be nuanced enough to represent that change as a very slightly different point in space.

The **attention mechanism** [introduced by Bahdanau et al.](https://arxiv.org/abs/1409.0473) addresses this by giving the decoder a way to "pay attention" to parts of the input, rather than relying on a single vector. For every step the decoder can select a different part of the input sentence to consider.

![](https://i.imgur.com/5y6SCvU.png)

Attention is calculated with another feedforward layer in the decoder. This layer will use the current input and hidden state to create a new vector, which is the same size as the input sequence (in practice, a fixed maximum length). This vector is processed through softmax to create *attention weights*, which are multiplied by the encoders' outputs to create a new context vector, which is then used to predict the next output.

![](https://i.imgur.com/K1qMPxs.png)

# Requirements

You will need [PyTorch](http://pytorch.org/) to build and train the models, and [matplotlib](https://matplotlib.org/) for plotting training and visualizing attention outputs later.


```python
import unicodedata
import string
import re
import random
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
```

Here we will also define a constant to decide whether to use the GPU (with CUDA specifically) or the CPU. **If you don't have a GPU, set this to `False`**. Later when we create tensors, this variable will be used to decide whether we keep them on CPU or move them to GPU.


```python
USE_CUDA = True
```

# Loading data files

The data for this project is a set of many thousands of English to French translation pairs.

[This question on Open Data Stack Exchange](http://opendata.stackexchange.com/questions/3888/dataset-of-sentences-translated-into-many-languages) pointed me to the open translation site http://tatoeba.org/ which has downloads available at http://tatoeba.org/eng/downloads - and better yet, someone did the extra work of splitting language pairs into individual text files here: http://www.manythings.org/anki/

The English to French pairs are too big to include in the repo, so download `fra-eng.zip`, extract the text file in there, and rename it to `data/eng-fra.txt` before continuing (for some reason the zipfile is named backwards). The file is a tab separated list of translation pairs:

```
I am cold.    Je suis froid.
```

Similar to the character encoding used in the character-level RNN tutorials, we will be representing each word in a language as a one-hot vector, or giant vector of zeros except for a single one (at the index of the word). Compared to the dozens of characters that might exist in a language, there are many many more words, so the encoding vector is much larger. We will however cheat a bit and trim the data to only use a few thousand words per language.

### Indexing words

We'll need a unique index per word to use as the inputs and targets of the networks later. To keep track of all this we will use a helper class called `Lang` which has word &rarr; index (`word2index`) and index &rarr; word (`index2word`) dictionaries, as well as a count of each word `word2count` to use to later replace rare words.


```python
SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2 # Count SOS and EOS
      
    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
```

### Reading and decoding files

The files are all in Unicode, to simplify we will turn Unicode characters to ASCII, make everything lowercase, and trim most punctuation.


```python
# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s
```

To read the data file we will split the file into lines, and then split lines into pairs. The files are all English &rarr; Other Language, so if we want to translate from Other Language &rarr; English I added the `reverse` flag to reverse the pairs.


```python
def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('../data/%s-%s.txt' % (lang1, lang2)).read().strip().split('\n')
    
    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
        
    return input_lang, output_lang, pairs
```

### Filtering sentences

Since there are a *lot* of example sentences and we want to train something quickly, we'll trim the data set to only relatively short and simple sentences. Here the maximum length is 10 words (that includes punctuation) and we're filtering to sentences that translate to the form "I am" or "He is" etc. (accounting for apostrophes being removed).


```python
MAX_LENGTH = 10

good_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re "
)

def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(good_prefixes)

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]
```

The full process for preparing the data is:

* Read text file and split into lines, split lines into pairs
* Normalize text, filter by length and content
* Make word lists from sentences in pairs


```python
def prepare_data(lang1_name, lang2_name, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1_name, lang2_name, reverse)
    print("Read %s sentence pairs" % len(pairs))
    
    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    
    print("Indexing words...")
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])

    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)

# Print an example pair
print(random.choice(pairs))
```

    Reading lines...
    Read 135842 sentence pairs
    Trimmed to 9129 sentence pairs
    Indexing words...
    ['il est paresseux .', 'he s lazy .']


## Turning training data into Tensors/Variables

To train we need to turn the sentences into something the neural network can understand, which of course means numbers. Each sentence will be split into words and turned into a Tensor, where each word is replaced with the index (from the Lang indexes made earlier). While creating these tensors we will also append the EOS token to signal that the sentence is over.

![](https://i.imgur.com/LzocpGH.png)

A Tensor is a multi-dimensional array of numbers, defined with some type e.g. FloatTensor or LongTensor. In this case we'll be using LongTensor to represent an array of integer indexes.

Trainable PyTorch modules take Variables as input, rather than plain Tensors. A Variable is basically a Tensor that is able to keep track of the graph state, which is what makes autograd (automatic calculation of backwards gradients) possible.


```python
# Return a list of indexes, one for each word in the sentence
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
#     print('var =', var)
    if USE_CUDA: var = var.cuda()
    return var

def variables_from_pair(pair):
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])
    return (input_variable, target_variable)
```

# Building the models

## The Encoder

<img src="images/encoder-network.png" style="float: right" />

The encoder of a seq2seq network is a RNN that outputs some value for every word from the input sentence. For every input word the encoder outputs a vector and a hidden state, and uses the hidden state for the next input word.


```python
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        
    def forward(self, word_inputs, hidden):
        # Note: we run this all at once (over the whole input sequence)
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA: hidden = hidden.cuda()
        return hidden
```

## Attention Decoder

### Interpreting the Bahdanau et al. model

The attention model in [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) is described as the following series of equations.

Each decoder output is conditioned on the previous outputs and some <img src="svgs/6ce54f9ab6efe2d0dcf7b19121f427db.svg?invert_in_darkmode" align=middle width=9.939930000000004pt height=14.55728999999999pt/>, where <img src="svgs/6ce54f9ab6efe2d0dcf7b19121f427db.svg?invert_in_darkmode" align=middle width=9.939930000000004pt height=14.55728999999999pt/> consists of the current hidden state (which takes into account previous outputs) and the attention "context", which is calculated below. The function <img src="svgs/3cf4fbd05970446973fc3d9fa3fe3c41.svg?invert_in_darkmode" align=middle width=8.398995000000005pt height=14.102549999999994pt/> is a fully-connected layer with a nonlinear activation, which takes as input the values <img src="svgs/dd5d0370a9a9f7b1c26ccf3ce0e8e61e.svg?invert_in_darkmode" align=middle width=29.426595000000002pt height=14.102549999999994pt/>, <img src="svgs/4fa3ac8fe93c68be3fe7ab53bdeb2efa.svg?invert_in_darkmode" align=middle width=12.310320000000004pt height=14.102549999999994pt/>, and <img src="svgs/3bc6fc8b86b6c61889f4e572c7546b8e.svg?invert_in_darkmode" align=middle width=11.720775000000003pt height=14.102549999999994pt/> concatenated.

<p align="center"><img src="svgs/8f8f44d3a200834ebd4dea6bbc5a6735.svg?invert_in_darkmode" align=middle width=269.34929999999997pt height=16.376943pt/></p>

The current hidden state <img src="svgs/4fa3ac8fe93c68be3fe7ab53bdeb2efa.svg?invert_in_darkmode" align=middle width=12.310320000000004pt height=14.102549999999994pt/> is calculated by an RNN <img src="svgs/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.780705000000003pt height=22.745910000000016pt/> with the last hidden state <img src="svgs/2f91f5c3839c1086a7f1d77e0eca8971.svg?invert_in_darkmode" align=middle width=29.073990000000002pt height=14.102549999999994pt/>, last decoder output value <img src="svgs/dd5d0370a9a9f7b1c26ccf3ce0e8e61e.svg?invert_in_darkmode" align=middle width=29.426595000000002pt height=14.102549999999994pt/>, and context vector <img src="svgs/3bc6fc8b86b6c61889f4e572c7546b8e.svg?invert_in_darkmode" align=middle width=11.720775000000003pt height=14.102549999999994pt/>.

In the code, the RNN will be a `nn.GRU` layer, the hidden state <img src="svgs/4fa3ac8fe93c68be3fe7ab53bdeb2efa.svg?invert_in_darkmode" align=middle width=12.310320000000004pt height=14.102549999999994pt/> will be called `hidden`, the output <img src="svgs/2b442e3e088d1b744730822d18e7aa21.svg?invert_in_darkmode" align=middle width=12.662925000000003pt height=14.102549999999994pt/> called `output`, and context <img src="svgs/3bc6fc8b86b6c61889f4e572c7546b8e.svg?invert_in_darkmode" align=middle width=11.720775000000003pt height=14.102549999999994pt/> called `context`.

<p align="center"><img src="svgs/011b229903a8612a43049525baee5a62.svg?invert_in_darkmode" align=middle width=144.98665499999998pt height=16.376943pt/></p>

The context vector <img src="svgs/3bc6fc8b86b6c61889f4e572c7546b8e.svg?invert_in_darkmode" align=middle width=11.720775000000003pt height=14.102549999999994pt/> is a weighted sum of all encoder outputs, where each weight <img src="svgs/db0dce2a6a38aedb28d33f6650cb22e8.svg?invert_in_darkmode" align=middle width=19.37199pt height=14.102549999999994pt/> is the amount of "attention" paid to the corresponding encoder output <img src="svgs/6d22be1359e204374e6f0b45e318d561.svg?invert_in_darkmode" align=middle width=15.517425000000003pt height=22.745910000000016pt/>.

<p align="center"><img src="svgs/a39ede7866ae8e0b8976fa9138cb619d.svg?invert_in_darkmode" align=middle width=96.66525pt height=50.188545pt/></p>

... where each weight <img src="svgs/db0dce2a6a38aedb28d33f6650cb22e8.svg?invert_in_darkmode" align=middle width=19.37199pt height=14.102549999999994pt/> is a normalized (over all steps) attention "energy" <img src="svgs/fffedfcb07fcd30112aa81594cf18315.svg?invert_in_darkmode" align=middle width=18.34074pt height=14.102549999999994pt/> ...

<p align="center"><img src="svgs/c589b88bfcfbc9a07d96f091ae5794cd.svg?invert_in_darkmode" align=middle width=147.397635pt height=42.654644999999995pt/></p>

... where each attention energy is calculated with some function <img src="svgs/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode" align=middle width=8.656725000000002pt height=14.102549999999994pt/> (such as another linear layer) using the last hidden state <img src="svgs/2f91f5c3839c1086a7f1d77e0eca8971.svg?invert_in_darkmode" align=middle width=29.073990000000002pt height=14.102549999999994pt/> and that particular encoder output <img src="svgs/6d22be1359e204374e6f0b45e318d561.svg?invert_in_darkmode" align=middle width=15.517425000000003pt height=22.745910000000016pt/>:

<p align="center"><img src="svgs/dc9c0faa50be71bd1604de17855f27b4.svg?invert_in_darkmode" align=middle width=116.094pt height=16.97751pt/></p>

### Implementing the Bahdanau et al. model

In summary our decoder should consist of four main parts - an embedding layer turning an input word into a vector; a layer to calculate the attention energy per encoder output; a RNN layer; and an output layer.

The decoder's inputs are the last RNN hidden state <img src="svgs/2f91f5c3839c1086a7f1d77e0eca8971.svg?invert_in_darkmode" align=middle width=29.073990000000002pt height=14.102549999999994pt/>, last output <img src="svgs/dd5d0370a9a9f7b1c26ccf3ce0e8e61e.svg?invert_in_darkmode" align=middle width=29.426595000000002pt height=14.102549999999994pt/>, and all encoder outputs <img src="svgs/0b3477a37ba87273f014bf66e7443dfe.svg?invert_in_darkmode" align=middle width=16.145745pt height=22.745910000000016pt/>.

* embedding layer with inputs <img src="svgs/dd5d0370a9a9f7b1c26ccf3ce0e8e61e.svg?invert_in_darkmode" align=middle width=29.426595000000002pt height=14.102549999999994pt/>
    * `embedded = embedding(last_rnn_output)`
* attention layer <img src="svgs/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode" align=middle width=8.656725000000002pt height=14.102549999999994pt/> with inputs <img src="svgs/c7ad43816c00aa34231d154c109e5d61.svg?invert_in_darkmode" align=middle width=66.36465pt height=24.56552999999997pt/> and outputs <img src="svgs/fffedfcb07fcd30112aa81594cf18315.svg?invert_in_darkmode" align=middle width=18.34074pt height=14.102549999999994pt/>, normalized to create <img src="svgs/db0dce2a6a38aedb28d33f6650cb22e8.svg?invert_in_darkmode" align=middle width=19.37199pt height=14.102549999999994pt/>
    * `attn_energies[j] = attn_layer(last_hidden, encoder_outputs[j])`
    * `attn_weights = normalize(attn_energies)`
* context vector <img src="svgs/3bc6fc8b86b6c61889f4e572c7546b8e.svg?invert_in_darkmode" align=middle width=11.720775000000003pt height=14.102549999999994pt/> as an attention-weighted average of encoder outputs
    * `context = sum(attn_weights * encoder_outputs)`
* RNN layer(s) <img src="svgs/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.780705000000003pt height=22.745910000000016pt/> with inputs <img src="svgs/36ebcfcea4a543f17477923b2bb527b7.svg?invert_in_darkmode" align=middle width=100.18007999999999pt height=24.56552999999997pt/> and internal hidden state, outputting <img src="svgs/4fa3ac8fe93c68be3fe7ab53bdeb2efa.svg?invert_in_darkmode" align=middle width=12.310320000000004pt height=14.102549999999994pt/>
    * `rnn_input = concat(embedded, context)`
    * `rnn_output, rnn_hidden = rnn(rnn_input, last_hidden)`
* an output layer <img src="svgs/3cf4fbd05970446973fc3d9fa3fe3c41.svg?invert_in_darkmode" align=middle width=8.398995000000005pt height=14.102549999999994pt/> with inputs <img src="svgs/2586e864fc062a5e41dd863b4e5f9617.svg?invert_in_darkmode" align=middle width=83.353545pt height=24.56552999999997pt/>, outputting <img src="svgs/2b442e3e088d1b744730822d18e7aa21.svg?invert_in_darkmode" align=middle width=12.662925000000003pt height=14.102549999999994pt/>
    * `output = out(embedded, rnn_output, context)`


```python
class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        
        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = GeneralAttn(hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, word_input, last_hidden, encoder_outputs):
        # Note that we will only be running forward for a single decoder time step, but will use all encoder outputs
        
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N
        word_embedded = self.dropout(word_embedded)
        
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N
        
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        
        # Final output layer
        output = output.squeeze(0) # B x N
        output = F.log_softmax(self.out(torch.cat((output, context), 1)))
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights
```

### Interpreting the Luong et al. model(s)

[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) by Luong et al. describe a few more attention models that offer improvements and simplifications. They describe a few "global attention" models, the distinction between them being the way the attention scores are calculated.

The general form of the attention calculation relies on the target (decoder) side hidden state and corresponding source (encoder) side state, normalized over all states to get values summing to 1:

<p align="center"><img src="svgs/06306dddc9e87be52da3ffb3d78ad87c.svg?invert_in_darkmode" align=middle width=334.33619999999996pt height=41.551455pt/></p>

The specific "score" function that compares two states is either *dot*, a simple dot product between the states; *general*, a a dot product between the decoder hidden state and a linear transform of the encoder state; or *concat*, a dot product between a new parameter <img src="svgs/4cde4ee8b98d8da807085ec94cad3c2b.svg?invert_in_darkmode" align=middle width=15.042060000000003pt height=14.102549999999994pt/> and a linear transform of the states concatenated together.

<p align="center"><img src="svgs/3d0c9b79e74c06397a3d1d761a36ff2a.svg?invert_in_darkmode" align=middle width=293.3766pt height=68.9865pt/></p>

The modular definition of these scoring functions gives us an opportunity to build specific attention module that can switch between the different score methods. The input to this module is always the hidden state (of the decoder RNN) and set of encoder outputs.


```python
class Attn(nn.Module):
    def __init__(self, method, hidden_size, max_length=MAX_LENGTH):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(seq_len)) # B x 1 x S
        if USE_CUDA: attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)
    
    def score(self, hidden, encoder_output):
        
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy
```

Now we can build a decoder that plugs this Attn module in after the RNN to calculate attention weights, and apply those weights to the encoder outputs to get a context vector.


```python
class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        
        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)
        
        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)
    
    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N
        
        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N
        
        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights
```

### Testing the models

To make sure the Encoder and Decoder model are working (and working together) we'll do a quick test with fake word inputs:


```python
encoder_test = EncoderRNN(10, 10, 2)
decoder_test = AttnDecoderRNN('general', 10, 10, 2)
print(encoder_test)
print(decoder_test)

encoder_hidden = encoder_test.init_hidden()
word_input = Variable(torch.LongTensor([1, 2, 3]))
if USE_CUDA:
    encoder_test.cuda()
    word_input = word_input.cuda()
encoder_outputs, encoder_hidden = encoder_test(word_input, encoder_hidden)

word_inputs = Variable(torch.LongTensor([1, 2, 3]))
decoder_attns = torch.zeros(1, 3, 3)
decoder_hidden = encoder_hidden
decoder_context = Variable(torch.zeros(1, decoder_test.hidden_size))

if USE_CUDA:
    decoder_test.cuda()
    word_inputs = word_inputs.cuda()
    decoder_context = decoder_context.cuda()

for i in range(3):
    decoder_output, decoder_context, decoder_hidden, decoder_attn = decoder_test(word_inputs[i], decoder_context, decoder_hidden, encoder_outputs)
    print(decoder_output.size(), decoder_hidden.size(), decoder_attn.size())
    decoder_attns[0, i] = decoder_attn.squeeze(0).cpu().data
```

    EncoderRNN (
      (embedding): Embedding(10, 10)
      (gru): GRU(10, 10, num_layers=2)
    )
    AttnDecoderRNN (
      (embedding): Embedding(10, 10)
      (gru): GRU(20, 10, num_layers=2, dropout=0.1)
      (out): Linear (20 -> 10)
      (attn): Attn (
        (attn): Linear (10 -> 10)
      )
    )
    torch.Size([1, 10]) torch.Size([2, 1, 10]) torch.Size([1, 1, 3])
    torch.Size([1, 10]) torch.Size([2, 1, 10]) torch.Size([1, 1, 3])
    torch.Size([1, 10]) torch.Size([2, 1, 10]) torch.Size([1, 1, 3])


# Training

## Defining a training iteration

To train we first run the input sentence through the encoder word by word, and keep track of every output and the latest hidden state. Next the decoder is given the last hidden state of the decoder as its first hidden state, and the `<SOS>` token as its first input. From there we iterate to predict a next token from the decoder.

### Teacher Forcing and Scheduled Sampling

"Teacher Forcing", or maximum likelihood sampling, means using the real target outputs as each next input when training. The alternative is using the decoder's own guess as the next input. Using teacher forcing may cause the network to converge faster, but [when the trained network is exploited, it may exhibit instability](http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf).

You can observe outputs of teacher-forced networks that read with coherent grammar but wander far from the correct translation - you could think of it as having learned how to listen to the teacher's instructions, without learning how to venture out on its own.

The solution to the teacher-forcing "problem" is known as [Scheduled Sampling](https://arxiv.org/abs/1506.03099), which simply alternates between using the target values and predicted values when training. We will randomly choose to use teacher forcing with an if statement while training - sometimes we'll feed use real target as the input (ignoring the decoder's output), sometimes we'll use the decoder's output.


```python
teacher_forcing_ratio = 0.5
clip = 5.0

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    # Get size of input and target sentences
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # Run words through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
    
    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:
        
        # Teacher forcing: Use the ground-truth target as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output[0], target_variable[di])
            decoder_input = target_variable[di] # Next target is next input

    else:
        # Without teacher forcing: use network's own prediction as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output[0], target_variable[di])
            
            # Get most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            
            decoder_input = Variable(torch.LongTensor([[ni]])) # Chosen word is next input
            if USE_CUDA: decoder_input = decoder_input.cuda()

            # Stop at end of sentence (not necessary when using known targets)
            if ni == EOS_token: break

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.data[0] / target_length
```

Finally helper functions to print time elapsed and estimated time remaining, given the current time and progress.


```python
def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))
```

## Running training

With everything in place we can actually initialize a network and start training.

To start, we initialize models, optimizers, and a loss function (criterion).


```python
attn_model = 'general'
hidden_size = 500
n_layers = 2
dropout_p = 0.05

# Initialize models
encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers)
decoder = AttnDecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p)

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

# Initialize optimizers and criterion
learning_rate = 0.0001
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()
```

Then set up variables for plotting and tracking progress:


```python
# Configuring training
n_epochs = 50000
plot_every = 200
print_every = 1000

# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every
```

    Starting job 591f9f701438c4613b4c4dc7 at 2017-05-20 03:02:21


To actually train, we call the train function many times, printing a summary as we go.

*Note:* If you run this notebook you can train, interrupt the kernel, evaluate, and continue training later. You can comment out the lines above where the encoder and decoder are initialized (so they aren't reset) or simply run the notebook starting from the following cell.


```python
# Begin!
for epoch in range(1, n_epochs + 1):
    
    # Get training data for this cycle
    training_pair = variables_from_pair(random.choice(pairs))
    input_variable = training_pair[0]
    target_variable = training_pair[1]

    # Run the train function
    loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss

    if epoch == 0: continue

    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
        print(print_summary)

    if epoch % plot_every == 0:
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0
```

    [log] 0m 42s (1000) 1.7562
    0m 42s (- 35m 0s) (1000 2%) 3.2168
    [log] 1m 28s (2000) 3.4178
    1m 28s (- 35m 14s) (2000 4%) 2.8085
    [log] 2m 13s (3000) 1.9268
    2m 13s (- 34m 50s) (3000 6%) 2.6295
    [log] 2m 59s (4000) 3.5481
    2m 59s (- 34m 24s) (4000 8%) 2.5226
    [log] 3m 45s (5000) 2.1306
    3m 45s (- 33m 51s) (5000 10%) 2.3431
    [log] 4m 31s (6000) 2.4112
    4m 31s (- 33m 9s) (6000 12%) 2.2012
    [log] 5m 16s (7000) 2.0306
    5m 16s (- 32m 26s) (7000 14%) 2.1778
    [log] 6m 3s (8000) 0.7172
    6m 3s (- 31m 46s) (8000 16%) 2.0516
    [log] 6m 49s (9000) 0.9867
    6m 49s (- 31m 3s) (9000 18%) 1.9482
    [log] 7m 35s (10000) 0.7058
    7m 35s (- 30m 22s) (10000 20%) 1.8463
    [log] 8m 22s (11000) 4.0532
    8m 22s (- 29m 40s) (11000 22%) 1.8389
    [log] 9m 8s (12000) 0.7909
    9m 8s (- 28m 56s) (12000 24%) 1.7710
    [log] 9m 53s (13000) 3.5531
    9m 53s (- 28m 10s) (13000 26%) 1.6996
    [log] 10m 40s (14000) 0.3723
    10m 40s (- 27m 26s) (14000 28%) 1.6392
    [log] 11m 27s (15000) 1.0735
    11m 27s (- 26m 43s) (15000 30%) 1.5887
    [log] 12m 13s (16000) 0.0578
    12m 13s (- 25m 58s) (16000 32%) 1.5159
    [log] 12m 59s (17000) 1.4627
    12m 59s (- 25m 13s) (17000 34%) 1.4669
    [log] 13m 46s (18000) 2.7408
    13m 46s (- 24m 28s) (18000 36%) 1.3980
    [log] 14m 33s (19000) 0.3085
    14m 33s (- 23m 44s) (19000 38%) 1.3614
    [log] 15m 19s (20000) 0.8777
    15m 19s (- 22m 59s) (20000 40%) 1.3163
    [log] 16m 6s (21000) 1.8394
    16m 6s (- 22m 15s) (21000 42%) 1.2799
    [log] 16m 53s (22000) 0.8656
    16m 53s (- 21m 29s) (22000 44%) 1.2038
    [log] 17m 39s (23000) 3.5788
    17m 39s (- 20m 43s) (23000 46%) 1.1853
    [log] 18m 26s (24000) 1.3385
    18m 26s (- 19m 58s) (24000 48%) 1.1643
    [log] 19m 12s (25000) 0.0158
    19m 12s (- 19m 12s) (25000 50%) 1.1351
    [log] 19m 59s (26000) 0.7937
    19m 59s (- 18m 27s) (26000 52%) 1.1285
    [log] 20m 46s (27000) 1.3123
    20m 46s (- 17m 41s) (27000 54%) 1.0553
    [log] 21m 32s (28000) 1.6989
    21m 32s (- 16m 55s) (28000 56%) 1.0265
    [log] 22m 18s (29000) 2.2208
    22m 18s (- 16m 9s) (29000 57%) 0.9440
    [log] 23m 4s (30000) 0.1320
    23m 4s (- 15m 23s) (30000 60%) 0.9769
    [log] 23m 51s (31000) 0.0043
    23m 51s (- 14m 37s) (31000 62%) 0.9395
    [log] 24m 37s (32000) 0.0119
    24m 37s (- 13m 51s) (32000 64%) 0.8899
    [log] 25m 23s (33000) 0.2071
    25m 23s (- 13m 5s) (33000 66%) 0.9135
    [log] 26m 10s (34000) 0.0169
    26m 10s (- 12m 19s) (34000 68%) 0.8698
    [log] 26m 57s (35000) 0.7662
    26m 57s (- 11m 33s) (35000 70%) 0.8209
    [log] 27m 43s (36000) 0.1208
    27m 43s (- 10m 46s) (36000 72%) 0.7931
    [log] 28m 29s (37000) 0.3535
    28m 29s (- 10m 0s) (37000 74%) 0.7899
    [log] 29m 15s (38000) 1.3398
    29m 15s (- 9m 14s) (38000 76%) 0.7603
    [log] 30m 2s (39000) 0.0115
    30m 2s (- 8m 28s) (39000 78%) 0.7454
    [log] 30m 48s (40000) 0.2135
    30m 48s (- 7m 42s) (40000 80%) 0.6740
    [log] 31m 34s (41000) 1.1087
    31m 34s (- 6m 55s) (41000 82%) 0.6738
    [log] 32m 20s (42000) 0.0262
    32m 20s (- 6m 9s) (42000 84%) 0.6659
    [log] 33m 7s (43000) 1.2855
    33m 7s (- 5m 23s) (43000 86%) 0.7443
    [log] 33m 54s (44000) 0.0022
    33m 54s (- 4m 37s) (44000 88%) 0.6427
    [log] 34m 40s (45000) 0.5267
    34m 40s (- 3m 51s) (45000 90%) 0.6092
    [log] 35m 27s (46000) 0.0068
    35m 27s (- 3m 4s) (46000 92%) 0.6172
    [log] 36m 12s (47000) 0.5520
    36m 12s (- 2m 18s) (47000 94%) 0.6145
    [log] 36m 59s (48000) 0.0185
    36m 59s (- 1m 32s) (48000 96%) 0.5903
    [log] 37m 46s (49000) 0.0026
    37m 46s (- 0m 46s) (49000 98%) 0.6131
    [log] 38m 32s (50000) 0.0138
    38m 32s (- 0m 0s) (50000 100%) 0.5403


## Plotting training loss

Plotting is done with matplotlib, using the array `plot_losses` that was created while training.


```python
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
%matplotlib inline

def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

show_plot(plot_losses)
```


    <matplotlib.figure.Figure at 0x7fae93740828>



![png](output_46_1.png)


# Evaluating the network

Evaluation is mostly the same as training, but there are no targets. Instead we always feed the decoder's predictions back to itself. Every time it predicts a word, we add it to the output string. If it predicts the EOS token we stop there. We also store the decoder's attention outputs for each step to display later.


```python
def evaluate(sentence, max_length=MAX_LENGTH):
    input_variable = variable_from_sentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    
    # Run through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([[SOS_token]])) # SOS
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    decoder_hidden = encoder_hidden
    
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)
    
    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])
            
        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([[ni]]))
        if USE_CUDA: decoder_input = decoder_input.cuda()
    
    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]
```

We can evaluate random sentences from the training set and print out the input, target, and output to make some subjective quality judgements:


```python
def evaluate_randomly():
    pair = random.choice(pairs)
    
    output_words, decoder_attn = evaluate(pair[0])
    output_sentence = ' '.join(output_words)
    
    print('>', pair[0])
    print('=', pair[1])
    print('<', output_sentence)
    print('')
```


```python
evaluate_randomly()
```

    > je suis ambitieux .
    = i m ambitious .
    < i m ambitious . <EOS>
    


# Visualizing attention

A useful property of the attention mechanism is its highly interpretable outputs. Because it is used to weight specific encoder outputs of the input sequence, we can imagine looking where the network is focused most at each time step.

You could simply run `plt.matshow(attentions)` to see attention output displayed as a matrix, with the columns being input steps and rows being output steps:


```python
output_words, attentions = evaluate("je suis trop froid .")
plt.matshow(attentions.numpy())
```


![png](output_53_0.png)


For a better viewing experience we will do the extra work of adding axes and labels:


```python
def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()

def evaluate_and_show_attention(input_sentence):
    output_words, attentions = evaluate(input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)
```


```python
evaluate_and_show_attention("elle a cinq ans de moins que moi .")
```

    input = elle a cinq ans de moins que moi .
    output = she s five years younger than me . <EOS>



![png](output_56_1.png)



```python
evaluate_and_show_attention("elle est trop petit .")
```

    input = elle est trop petit .
    output = she s too short . <EOS>



![png](output_57_1.png)



```python
evaluate_and_show_attention("je ne crains pas de mourir .")
```

    input = je ne crains pas de mourir .
    output = i m not scared to die . <EOS>



![png](output_58_1.png)



```python
evaluate_and_show_attention("c est un jeune directeur plein de talent .")
```

    input = c est un jeune directeur plein de talent .
    output = he s a very young young . <EOS>



![png](output_59_1.png)


# Exercises

* Try with a different dataset
    * Another language pair
    * Human &rarr; Machine (e.g. IOT commands)
    * Chat &rarr; Response
    * Question &rarr; Answer
* Replace the embedding pre-trained word embeddings such as word2vec or GloVe
* Try with more layers, more hidden units, and more sentences. Compare the training time and results.
* If you use a translation file where pairs have two of the same phrase (`I am test \t I am test`), you can use this as an autoencoder. Try this:
    * Train as an autoencoder
    * Save only the Encoder network
    * Train a new Decoder for translation from there


```python

```
