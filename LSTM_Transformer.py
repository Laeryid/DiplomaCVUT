import tensorflow as tf
import tensorflow_addons as tfa

class ViTEmbeddings(tf.keras.layers.Layer):
    def __init__(self, patch_size, hidden_size, dropout=0.0, **kwargs):
        super().__init__(**kwargs)

        self.patch_size = patch_size
        self.hidden_size = hidden_size

        self.patch_embeddings = tf.keras.layers.Conv1D(filters=hidden_size, kernel_size=patch_size, strides=patch_size)
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def build(self, input_shape):
        self.cls_token = self.add_weight(shape=(1, 1, self.hidden_size), trainable=True, name="cls_token")

        num_patches = input_shape[1] // self.patch_size
        self.position_embeddings = self.add_weight(
            shape=(1, num_patches + 1, self.hidden_size), trainable=True, name="position_embeddings"
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        inputs_shape = tf.shape(inputs)  # N,H,W,C
        embeddings = self.patch_embeddings(inputs, training=training)

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = tf.repeat(self.cls_token, repeats=inputs_shape[0], axis=0)
        embeddings = tf.concat((cls_tokens, embeddings), axis=1)

        # add positional encoding to each token
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings, training=training)

        return embeddings

class MLP(tf.keras.layers.Layer):
    def __init__(self, mlp_dim, out_dim=None, activation="gelu", dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.mlp_dim = mlp_dim
        self.out_dim = out_dim
        self.activation = activation
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(self.mlp_dim)
        self.activation1 = tf.keras.layers.Activation(self.activation)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense2 = tf.keras.layers.Dense(input_shape[-1] if self.out_dim is None else self.out_dim)

    def call(self, inputs: tf.Tensor, training: bool = False):
        x = self.dense1(inputs)
        x = self.activation1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        return x

class Block(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        attention_dim,
        attention_bias,
        mlp_dim,
        attention_dropout=0.0,
        sd_survival_probability=1.0,
        activation="gelu",
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm_before = tf.keras.layers.LayerNormalization()
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads,
            attention_dim // num_heads,
            use_bias=attention_bias,
            dropout=attention_dropout,
        )
        self.stochastic_depth = tfa.layers.StochasticDepth(sd_survival_probability)
        self.norm_after = tf.keras.layers.LayerNormalization()
        self.mlp = MLP(mlp_dim=mlp_dim, activation=activation, dropout=dropout)

    def build(self, input_shape):
        super().build(input_shape)
        # TODO YONIGO: tf doc says to do this  ¯\_(ツ)_/¯
        self.attn._build_from_signature(input_shape, input_shape)

    def call(self, inputs, training=False):
        x = self.norm_before(inputs, training=training)
        x = self.attn(x, x, training=training)
        x = self.stochastic_depth([inputs, x], training=training)
        x2 = self.norm_after(x, training=training)
        x2 = self.mlp(x2, training=training)
        return self.stochastic_depth([x, x2], training=training)

    def get_attention_scores(self, inputs):
        x = self.norm_before(inputs, training=False)
        _, weights = self.attn(x, x, training=False, return_attention_scores=True)
        return weights

class VisionTransformer(tf.keras.Model):
    def __init__(
        self,
        patch_size,
        hidden_size,
        depth,
        num_heads,
        mlp_dim,
        num_classes,
        dropout=0.0,
        sd_survival_probability=1.0,
        attention_bias=False,
        attention_dropout=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.embeddings = ViTEmbeddings(patch_size, hidden_size, dropout)
        sd = tf.linspace(1.0, sd_survival_probability, depth)
        self.blocks = [
            Block(
                num_heads,
                attention_dim=hidden_size,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
                mlp_dim=mlp_dim,
                sd_survival_probability=(sd[i].numpy().item()),
                dropout=dropout,
            )
            for i in range(depth)
        ]

        self.norm = tf.keras.layers.LayerNormalization()

        self.head = tf.keras.layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.embeddings(inputs, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.norm(x)
        x = x[:, 0]  # take only cls_token
        return self.head(x)

    def get_last_selfattention(self, inputs: tf.Tensor):
        x = self.embeddings(inputs, training=False)
        for block in self.blocks[:-1]:
            x = block(x, training=False)
        return self.blocks[-1].get_attention_scores(x)

def get_Transformer_model(input_shape, n_classes):
    return VisionTransformer(
        patch_size=20,
        hidden_size=768,
        depth=6,
        num_heads=6,
        mlp_dim=256,
        num_classes=n_classes,
        sd_survival_probability=0.9,
        )

# First, let's define a RNN Cell, as a layer subclass.
class MinimalRNNCell(tf.keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = tf.keras.backend.dot(inputs, self.kernel)
        output = h + tf.keras.backend.dot(prev_output, self.recurrent_kernel)
        return output, [output]

def get_RNN_CNN_model(input_shape, n_classes):
    input = tf.keras.Input(shape=input_shape, name='CNN_input_X')
    biRNN = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(MinimalRNNCell(12*6), return_sequences=True))(input)
    CNN = tf.keras.layers.Conv1D(6, 3)(biRNN)
    Flat = tf.keras.layers.Flatten()(CNN)
    output = tf.keras.layers.Dense(n_classes, activation='sigmoid')(Flat)
    
    model = tf.keras.models.Model(inputs=input
                                      , outputs=output)
    return model

def get_RNN_model(input_shape, n_classes):
    input = tf.keras.Input(shape=input_shape, name='CNN_input_X')
    RNN = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(MinimalRNNCell(12*6)))(input)
    output = tf.keras.layers.Dense(n_classes, activation='sigmoid')(RNN)
    
    model = tf.keras.models.Model(inputs=input
                                        , outputs=output)
    return model

def get_LSTM_CNN_model(input_shape, n_classes):
    input = tf.keras.Input(shape=input_shape, name='CNN_input_X')
    biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(12*6, dropout=0.2, return_sequences=True))(input)
    CNN = tf.keras.layers.Conv1D(6, 3)(biLSTM)
    Flat = tf.keras.layers.Flatten()(CNN)
    output = tf.keras.layers.Dense(n_classes, activation='sigmoid')(Flat)
    
    model = tf.keras.models.Model(inputs=input
                                      , outputs=output)
    return model

def get_LSTM_model(input_shape, n_classes):
    input = tf.keras.Input(shape=input_shape, name='CNN_input_X')
    LSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(12*6, dropout=0.2))(input)
    output = tf.keras.layers.Dense(n_classes, activation='sigmoid')(LSTM)
    
    model = tf.keras.models.Model(inputs=input
                                        , outputs=output)
    return model