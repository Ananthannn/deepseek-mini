import tensorflow as tf
from .rope import RoPE


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, max_seq_len=2048, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = tf.math.sqrt(tf.cast(self.head_dim, tf.float32))

        # Projections
        self.qkv = tf.keras.layers.Dense(3 * embed_dim, use_bias=False)
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=False)

        # Dropout
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # RoPE
        self.rope = RoPE(self.head_dim, max_seq_len)

    def split_heads(self, x, B, T):
        # (B, T, C) → (B, H, T, D)
        x = tf.reshape(x, (B, T, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def combine_heads(self, x, B, T):
        # (B, H, T, D) → (B, T, C)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (B, T, self.embed_dim))

    def build_causal_mask(self, T):
        # (1, 1, T, T)
        mask = tf.linalg.band_part(tf.ones((T, T)), -1, 0)
        mask = tf.cast(mask, tf.bool)
        return mask[tf.newaxis, tf.newaxis, :, :]

    def call(self, x, attention_mask=None, training=False):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        NEG_INF = tf.constant(-1e9, dtype=x.dtype)

        # -----------------------------
        # QKV Projection
        # -----------------------------
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = tf.split(qkv, 3, axis=-1)

        # -----------------------------
        # Split Heads
        # -----------------------------
        q = self.split_heads(q, B, T)
        k = self.split_heads(k, B, T)
        v = self.split_heads(v, B, T)

        # -----------------------------
        # Apply RoPE
        # -----------------------------
        cos, sin = self.rope.get_cos_and_sine(T)
        q, k = self.rope.apply(q, k, cos, sin)

        # -----------------------------
        # Attention Scores
        # -----------------------------
        scores = tf.matmul(q, k, transpose_b=True)  # (B, H, T, T)
        scores = scores / self.scale

        # -----------------------------
        # Causal Mask
        # -----------------------------
        causal_mask = self.build_causal_mask(T)
        scores = tf.where(causal_mask, scores, NEG_INF)

        # -----------------------------
        # Padding Mask
        # -----------------------------
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, tf.bool)
            attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]  # (B,1,1,T)
            scores = tf.where(attention_mask, scores, NEG_INF)

        # -----------------------------
        # Softmax
        # -----------------------------
        attn_probs = tf.nn.softmax(scores, axis=-1)

        # -----------------------------
        # Dropout (important for training)
        # -----------------------------
        attn_probs = self.dropout(attn_probs, training=training)

        # -----------------------------
        # Weighted Sum
        # -----------------------------
        out = tf.matmul(attn_probs, v)  # (B, H, T, D)

        # -----------------------------
        # Combine Heads
        # -----------------------------
        out = self.combine_heads(out, B, T)

        # -----------------------------
        # Final Projection
        # -----------------------------
        out = self.out_proj(out)

        return out