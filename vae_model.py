import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):
  """
  Camada customizada para realizar a amostragem do espaço latente.
    **kwargs: Argumentos adicionais para a camada base.
    Retorna a amostra do espaço latente.
  """
  def __init__(self, **kwargs):
    super(Sampling, self).__init__(**kwargs)

  def call(self, inputs):
    """
    Realiza a amostragem do espaço latente.
        inputs: Uma lista contendo a média e a variância do espaço latente.
        Retorna a amostra do espaço latente.
    """
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def create_encoder(input_shape, latent_dim, summary=False):
  """
  Cria o encoder da rede VAE.
    input_shape: a forma da entrada da rede (altura, largura, canais de cor).
    latent_dim: a dimensionalidade do espaço latente.
    summary: verdadeiro ou falso para printar o summary.
    É retornado modelo Keras representando o encoder.
  """

  encoder_inputs = keras.Input(shape=input_shape)

  # Camada convolucional onde será dado o input
  # Para as camadas, são aplicados o BatchNormalization e a função de ativaçao Leaky ReLU
  
  body = layers.Conv2D(64, 3, strides=2, padding="same")(encoder_inputs)
  body = layers.BatchNormalization()(body)
  body = layers.LeakyReLU(0.2)(body)

  # Bloco de camadas convolucionais para extrair caracteristicas
  for filter_size in [128, 128, 128]:
    body = layers.Conv2D(filter_size, 3, strides=2, padding="same")(body)
    body = layers.BatchNormalization()(body)
    body = layers.LeakyReLU(0.2)(body)

  # Convertendo para vetor
  body = layers.Flatten()(body)

  # Camada densa com 2048 neurônios
  body = layers.Dense(2048)(body)
  body = layers.BatchNormalization()(body)
  body = layers.LeakyReLU(0.2)(body)

  # Camadas para obter a média e variância do espaço latente
  z_mean = layers.Dense(latent_dim, name="z_mean")(body)
  z_log_var = layers.Dense(latent_dim, name="z_log_var")(body)

  # Camada Sampling para amostrar do espaço latente
  z = Sampling()([z_mean, z_log_var])

  # Cria o modelo do encoder
  encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="Encoder")

  if summary:
    encoder.summary()

  return encoder


def create_decoder(latent_dim, summary=False):
  """
  Cria o decoder da rede VAE.
    latent_dim: A dimensionalidade do espaço latente.
    summary: verdadeiro ou falso para printar o summary.

    É retornado modelo Keras representando o decoder.
  """

  latent_inputs = keras.Input(shape=(latent_dim,))


  # Camadas densa para aumentar a dimensionalidade da saída
  # Camada densa com 2048 neuronios
  body = layers.Dense(2048)(latent_inputs)
  body = layers.BatchNormalization()(body)
  body = layers.LeakyReLU(0.2)(body)
  
  # Camada densa com 16384 neurônios
  body = layers.Dense(16384)(body)
  body = layers.BatchNormalization()(body)
  body = layers.LeakyReLU(0.2)(body)

  # Camada para 'remodelar' o vetor de alta dimensão em um tensor 3D
  body = layers.Reshape((8, 8, 256))(body)

  # Bloco de camadas para reconstruir a imagem. Diminuir o filter)size
  # ajuda a controlar a complexidade dos recursos que a camada pode aprender

  for filter_size in [128, 64, 32]:
    body = layers.Conv2DTranspose(filter_size, 3, strides=2, padding="same")(body)
    body = layers.BatchNormalization()(body)
    body = layers.LeakyReLU(0.2)(body)

  # Camada final com saida de 3 canais (RGB) e ativação sigmoid
  decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(body)

  # Cria o modelo do decoder
  decoder = keras.Model(latent_inputs, decoder_outputs, name="Decoder")

  if summary:
    decoder.summary()

  return decoder


def normalize_image(batch):
  """
  Normaliza a imagem dividindo por 255.
    batch: Um tensor contendo as imagens.
    É retornado tensor com as imagens normalizadas.
  """
  batch = batch / 255.
  return batch


class VAE(keras.Model):
  """
  Modelo VAE completo.
  
    encoder: O encoder da rede VAE.
    decoder: O decoder da rede VAE.
    **kwargs: Argumentos adicionais para o modelo base.
    total_loss_tracker: Uma métrica para acompanhar a perda total.
    reconstruction_loss_tracker: Uma métrica para acompanhar a perda de reconstrução.
    kl_loss_tracker: Uma métrica para acompanhar a perda de KL.
  """

  def __init__(self, encoder, decoder, **kwargs):
    super(VAE, self).__init__(**kwargs)
    self.encoder = encoder
    self.decoder = decoder

    # Métricas para acompanhar o treinamento
    self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
    self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
    self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

  @property
  def metrics(self):
    """
    Retorna as métricas utilizadas para acompanhar o treinamento.
    (lista contendo as métricas).
    """
    return [
      self.total_loss_tracker,
      self.reconstruction_loss_tracker,
      self.kl_loss_tracker,
    ]

  def train_step(self, data):
    """
    Realiza um passo de treinamento da rede VAE.
        data: Um tensor contendo as imagens de entrada.
        é retornado um dicionário com as perdas do treinamento.
    """

    with tf.GradientTape() as tape:
      z_mean, z_log_var, z = self.encoder(data)
      reconstruction = self.decoder(z)

      # Perda durante a reconstrução
      reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(
          keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
        )
      )

      # Perda de KL (Divergência de Kullback-Leibler)
      # importante para incentivar a geração latente mais próxima da de uma gaussiana
      kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
      kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

      # Quantificando a perda total
      total_loss = reconstruction_loss + kl_loss

    grads = tape.gradient(total_loss, self.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

    # Atualiza as métricas
    self.total_loss_tracker.update_state(total_loss)
    self.reconstruction_loss_tracker.update_state(reconstruction_loss)
    self.kl_loss_tracker.update_state(kl_loss)

    return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
  
def build_model(save_model: bool = False,
                image_size: int = 128,
                latent_dim: int = 128,
                batch_size: int = 32,
                n_epochs: int = 10,
                data_path: object = 'data/train/',
                summary=False):

  """
  Constrói e treina o modelo VAE.

    save_model: verdadeiro ou falso se deve salvar os modelos encoder e decoder.
    image_size: A altura e largura das imagens.
    latent_dim: A dimensionalidade do espaço latente.
    batch_size: O tamanho do lote para o treinamento.
    n_epochs: O número de épocas para o treinamento.
    data_path: O caminho para o diretório contendo as imagens de treinamento.
    summary: verdadeiro ou falso para printar o summary.

    É retornado o modelo VAE treinado.
  """
  
  input_shape = (image_size, image_size, 3)

  # Cria o dataset de imagens
  batch_dataset = tf.keras.utils.image_dataset_from_directory(
    data_path,
    label_mode = None,
    seed=42,
    image_size=(image_size, image_size),
    batch_size=batch_size
  )

  # Normaliza o dataset
  batch_dataset_norm = batch_dataset.map(normalize_image)

  # Cria o encoder e decoder
  encoder = create_encoder(input_shape, latent_dim, summary)
  decoder = create_decoder(latent_dim, summary)

  # Cria o modelo VAE
  vae = VAE(encoder, decoder)

  # Compila o modelo com otimizador Adam e lr
  vae.compile(optimizer=keras.optimizers.Adam(1e-4))

  # Realiza o treinamento
  vae.fit(batch_dataset_norm, epochs=n_epochs)

  # Salva os modelos se necessário
  if save_model:
    vae.decoder.save('vae_dog_decoder')
    vae.encoder.save('vae_dog_encoder')

  return vae


if __name__ == '__main__':
  build_model(n_epochs=50, save_model=True)
