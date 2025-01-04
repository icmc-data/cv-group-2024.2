# Abordagem com MediaPipe

## Sobre

O projeto propõe a realização da classificação de palavras da linguagem brasileira de sinais (LIBRAS) com base em imagens de um dataset. Para tanto, o grupo utilizou diferentes abordagens, dentre as quais está a detecção de sinais das mãos utilizando Mediapipe para coletar os keypoints (pontos-chave). São utilizados 21 pontos, que possuem coordenadas x, y, z. Cada uma dessas coordenadas foram armazenadas em um arquivo csv

## Objetivos e Ferramentas

O processo de desenvolvimento do projeto para esta abordagem tem 2 pontos principais:

- Utilizar MediaPipe para detectar a mão e o esqueleto da mesma nas imagens de treino do dataset, bem como seus keypoints e, após, coletar as coordenadas dos keypoints das maõs e armazena-las em um arquivo csv chamado handgestures.csv

- Utilizar um modelo de Machine Learning chamado Random Forests para classificar o sinal das maõs utilizando como dados as coordenadas x, y e z de cada um dos pontos e a classe representada pela mão.

Para o primeiro objetivo, foi utilizado o dataset presente no seguinte link: https://universe.roboflow.com/gomes-project/projeto-libras/dataset/21. Nele, temos conjuntos de imagens para treino, teste e validação, as quais estão divididas em 35 classes. Após isso, foi utilizada a biblioteca MediaPipe para, em cada imagem de treino, detectar as mãos e seus respectivos keypoints e armazenar as coordenadas no arquivo csv. Também utilizei outras bibliotecas, como OpenCV e CSV.

Para a etapa do modelo RandomForestClassifier, utilizei a biblioteca SKLEARN para fazer o treinamento e teste. Mas antes disso, fiz a leitura dos dados no arquivo handgestures.csv, para que fossem utilizados no treinamento. Separei 30% deles para teste, enquanto que os outros 70% foram utilizados no treinamento. Por fim, usei a biblipteca Pickle para salvar o modelo.

## Arquivos

- **libras_com_mediapipe.ipynb**: baixa as imagens do dataset, aplica o mediapipe e coleta as coordenadas

- **treinamento_modelo.ipynb**: faz o treinamento do modelo RandomForestClassifier e o salva para ser utilizado com a webcam

- **webcam.py**: carrega o modelo modelo_sign_language.pkl e testa o modelo em tempo real por meio da webcam
## Resultado

Ao final das etapas de processamento dos dados e treinamento/teste do modelo de aprendizado de máquina, obteve-se uma precisão média de 80% sobre os dados de teste. Entretanto, vale ressaltar que, ao combinar os dados das coordenadas obtidas nas imagens do dateset com dados próprios, ou seja, obtidos por meio da sua própria webcam, o resultado fica melhor, obtendo 99,9% de precisão média sobre os dados de teste. Isso ocorre, pois a quantidade de dados fica extremamente maior, devido à grande quantidade de frames ao ligar a webcam, além de uma maior possibilidade de alterar ângulos do sinal.


