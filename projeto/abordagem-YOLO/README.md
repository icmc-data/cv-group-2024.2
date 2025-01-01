# Abordagem 1 - YOLO
## Sobre
O projeto propõe a classificação de palavras da linguagem brasileira de sinais (LIBRAS) em imagens por meio de um modelo de rede neural. Para tanto, o grupo utilizou diferentes abordagens, dentre as quais está a utilização de um modelo YOLO como base para um modelo capaz de identificar, enquadrar e classificar sinais de LIBRAS.
## Objetivos e Ferramentas
O processo de desenvolvimento do projeto se baseou na escolha de um modelo YOLO base e no subsequente treinamento e teste de nosso modelo especializado.
De inicio, foi escolhido o modelo YOLOv11n, a versão mais simples do modelo mais recente. A partir dai, utilizamos o Google Collab para treinar o modelo com base no dataset presente no seguinte link: https://universe.roboflow.com/gomes-project/projeto-libras/dataset/21. Nele, temos conjuntos de imagens para treino, teste e validação, as quais estão divididas em 35 classes.
Por fim, partimos para os testes do modelo utilizando a função de avaliação do YOLO e um código de análise em tempo real pela webcam.
## Arquivos
- webcam.py - Código python utilizado para testar o algoritmo em tempo real por meio da webcam de um computador pessoal;
- libras.ipynb - Jupyter Notebook utilizado no Google Collab para treinar o modelo;
- best-YOLOn.pt - Arquivo contendo o modelo treinado.
## Resultados
Ao final das etapas de processamento dos dados e treinamento/teste do modelo de aprendizado de máquina, obteve-se uma precisão média de 97% sobre o dataset de teste, o que representa um resultado positivo ao comportamento do modelo. Além disso, foi percebido um bom desempenho nos testes ao vivo.
