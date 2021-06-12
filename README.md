# transformer-exploration
Exploring Transformer-based architectures as a final project for CS321 Advanced Machine Learning.

### Resources
- https://arxiv.org/abs/1706.03762
  -  The original paper on the Transformer model, entitled "Attention Is All You Need."
- https://jalammar.github.io/illustrated-transformer/
  - A well-explained paper illustrating the inner mechanisms behind the core ideas of the Transformer model.
- https://www.youtube.com/watch?v=rBCqOTEfxvg
- https://colah.github.io/posts/2015-09-Visual-Information/
  - A really helpful article explaining the ideas of cross-entropy and KL divergence in a clear and easy-to-access way.
- https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained
  - Another article explaining what KL divergence is.
- https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms
  - A nice Stats Stack Exchange answer explaining what QKV (query-key-value) is in attention mechanisms.

## The Transformer

Transformers have a variety of different applications throughout natural language processing—from speech recognition to machine translation to text summarization, this technology has seen widespread uptake across the world of NLP.

Transformers consist of two subcomponents—the **encoder** and **decoder**. Input words are converted to word vectors of fixed dimension using some kind of word-embedding algorithm (such as FastText or word2vec), and are then fed into the encoder. Below is a simplified description of how the Transformer actually works—some implementation details have been overlooked.

### The Encoder

The first and most important part of the encoder is the **self-attention** layer, which consists of a **query, key, value** formulation commonly found in other information retrieval mechanisms. Each word vector in the input sentence is fed into an *attention head*. The word vector is multiplied by query, key, and value matrices (`W_q`, `W_k`, and `W_v`) to produce query, key, and value vectors `q_i`, `k_i`, and `v_i` (which usually have a lower dimension than the word vectors). Subsequently, each word is assigned a scaled score `s_i = (q_i · k_i)/sqrt(d_k)` (where `d_k` is the dimension of the key vector). All of the `s_i`'s are scaled using softmax, producing `S_i`, then the output of the self-attention head `z_i` is the sum of all of the `S_i * v_i`. 

Within a Transformer model, there are usually multiple *attention heads*, which all have their own query, key, and value matrices as described above. In order to produce a single output per word, all of the attention head outputs `z_i,j` are concatenated into one large matrix `Z_i`, which is then multiplied by a weight matrix trained simultaneously with the model to produce the final output `Z_i`. These `Z_i` vectors are then fed into a traditional feed-forward neural network to produce the output for a single encoder, which is then fed into the next encoder in the stack.

Because this implementation doesn't actually account for the *position* of each word in the sentence (there is no intrinsic ordering within any of these operations), **positional encoding** is used. Each word vector has a trained positional encoding vector (of the same dimension) added to it before being input into the encoder's initial self-attention layer.

### The Decoder

Each decoder consists of three layers—a **self-attention** layer, an **encoder-decoder attention** layer (which helps the decoders target specific sections of the input sequence), and a **feed-forward neural network**. This high-level architecture is identical to the encoder except for the presence of an extra attention layer in the middle.

There are some differences in implementation between the encoders and decoders, however. The self-attention layer is also prohibited from seeing words/vectors at "future" positions in the output sequence. The output of the final encoder in the original stack is converted into a set of key and value vectors, which are used in each encoder-decoder attention layer. The query vectors/matrices are determined during training.

Because the decoder stack outputs a vector for each word (just like the encoder stack), this vector is converted into a single word by using a feed-forward neural network with a final softmax activation function to produce probabilities for each word in the vocabulary.