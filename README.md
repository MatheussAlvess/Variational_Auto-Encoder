# Variational AutoEncoder para Imagens de Cachorros

## O projeto consiste em construir uma Arquitetura VAE (Variational-AutoEncoder) para aprender características de imagens de cachorros e ser capaz de reconstruí-las.

## Conjunto de dados:

Os dados consistem em imagens de cachorros, as quais são armazenadas nas pastas `train` e `test`. Existe também um `.csv` que associa o nome da imagem com a raça do cachorro.
Estes dadps podem ser obtidos via Kaggle: [Dog Breed Indetification](https://www.kaggle.com/c/dog-breed-identification/data).

## Conteúdo:

No repositório podem ser encontrados os arquivos:
- `vae_model.py` -> Aqui é construída arquitetura de encoder decoder utilizada, além de funções auxiliares.
- `tasks.py` - > Aqui são respondidas as questões a respeito de Missigenação e Identificação de raça.
- `requirements.txt` -> Dependências.
- Complementarmente, temos a pasta `images` contendo as imagens utilizadas na execução das tasks. 

## Uso:

Caso haja o interesse em executar os códigos e replicar os outputs, basta executar `$python vae_model.py`. Dessa forma, serão salvos tanto o encoder quanto o decoder com os parâmetros _default_ definidos na função `build_model`.

### Para executar os códigos e replicar os resultados, siga estas etapas:

1. Clone este repositório para o seu computador utilizando o comando no terminal:
```
git clone https://github.com/MatheussAlvess/Variational_AutoEncoder.git
```
2. Navegue até o diretório do projeto.
3. Garanta ter as dependências necessárias executando no terminal:
```
 pip install -r requirements.txt
```
5. Descompacte a pasta do dataset e renomeie para 'data'. (Está pasta deve conter a subpasta _train_)
6. Execute o seguinte comando no terminal:
   `python vae_model.py`
   
#### O que o comando faz?

- O comando executará o script `vae_model.py`.
- Este script carrega os dados, treina o modelo VAE e salva os modelos encoder e decoder.
- Os modelos são salvos no diretório atual com os parâmetros _default_ definidos na função `build_model`.

#### Observações:
- Você pode modificar os parâmetros do modelo VAE editando o script `vae_model.py`, tanto passando os parâmetros para o `build_model` quanto variando internamente os hiperparâmetros da arquitetura (como o tamanho dos filtros, quantidade de camadas, etc.).
- Esta é a primeira versão do projeto, ou seja, ainda há muitas melhorias que podem ser feitas. Isso inclui a otimização de hiperparâmetros e estrutura das arquiteturas encoder decoder.
- A arquitetura foi fundamentada no [ _code examples_](https://keras.io/examples/generative/vae/) do Keras, que utiliza os dados MNIST, visando ter um ponto de partida "validado", com a intenção de ter resultados eficiente com brevidade.
- Os resultados foram aceitáveis, uma vez que nada foi otimizado. Mas para melhorar o desempenho, recomento aumentar o número de épocas, avaliar as métricas de perda, variar a estrutura tanto do encoder quanto do decoder além de variar os parâmetros no geral. 

#### Próximos passos:
- Realizar clusterização no espaço latente considerando as raças. Tendo como objetivo conseguir avaliar, dado uma imagem de input, qual a raça do cachorro.

> [!NOTE]
> Devido o tamanho dos modelos salvos, não foi possível subir no repositório. Para replicar os resultados, basta executar com os parâmetros _default_.
