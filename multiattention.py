import tensorflow as tf
import numpy as np
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self):

        super(MultiHeadAttention, self).__init__()
        # self.ouput_dim = args.task_num
        self.d_k = self.d_v = 32  #128/head
        self.W_Q = tf.keras.layers.Dense(128, use_bias=False)
        self.W_K = tf.keras.layers.Dense(128, use_bias=False)
        self.W_V = tf.keras.layers.Dense(128, use_bias=False)
        self.fc = tf.keras.layers.Dense(128, use_bias=False)

    # def forward(self, X):
    #     ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
    #     Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
    #     K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
    #     V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)
    #
    #     scores = tf.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
    #     # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
    #     attn = tf.nn.softmax(dim=-1)(scores)
    #     context = tf.matmul(attn, V)
    #     # context: [len_q, n_heads * d_v]
    #     context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
    #     output = self.fc(context)
    #     return output

    # 这种情形下就是交叉注意力的形式。输入的是两个不同的特征向量
    def forward(self, X1, X2):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X1).reshape(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X2).reshape(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X2).reshape(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = tf.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = tf.nn.softmax(dim=-1)(scores)
        context = tf.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        # output = self.AN1(output)
        # output = self.l1(output)
        return output
