# Abordagem com CNN e Detecção de Objetos

## Sobre

O projeto propõe a realização da classificação de palavras da linguagem brasileira de sinais (LIBRAS) com base em imagens de um dataset. Para tanto, o grupo utilizou diferentes abordagens, dentre as quais está a detecção de mãos a partir de fotos e o desenvolvimento de uma rede neural para classificação.

## Objetivos e Ferramentas

O processo de desenvolvimento do projeto para esta abordagem tem 2 pontos principais:

- Utilizar uma técnica de detecção e recorte de mãos em imagens de um dataset

- Construir, treinar e testar uma rede neural CNN para classificação das imagens geradas pela etapa anterior

Para o primeiro objetivo, foi utilizado o dataset presente no seguinte link: https://universe.roboflow.com/gomes-project/projeto-libras/dataset/21. Nele, temos conjuntos de imagens para treino, teste e validação, as quais estão divididas em 35 classes. Para a detecção, utilizamos o framework CVZone (https://github.com/cvzone/cvzone) junto a bibliotecas tradicionais de Python, como Numpy, Pandas e OpenCV.

Para a etapa da rede neural, utilizamos o ambiente do Google Colab para executar os processos necessários. Com o auxílio da biblioteca Pytorch, construímos uma CNN e realizamos seu treinamento e teste.

## Arquivos

- **preProcessing.py**: itera sobre as imagens do dataset, realiza os ajustes necessários e cria novas imagens.

- **handDetection.py**: toma as imagens passadas pelo preProcessing.py e determina, a partir de um algoritmo do CVZone, as coordenadas do bounding box que comporta as mãos.

- **signLanguageModel.ipynb**: manipula o dataset e desenvolve uma rede neural.

## Resultado

Ao final das etapas de processamento dos dados e treinamento/teste do modelo de aprendizado de máquina, obteve-se uma precisão média de 95% sobre o dataset de teste, o que representa um resultado positivo ao comportamento do modelo.


