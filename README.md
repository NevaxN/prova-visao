# prova-visao

1 - Descrição do Problema
R:
O problema 1 tem como ideia aplicar um pré-processamento onde eu redimensiono as imagens e aplico filtros como GaussianBlur e Equalização de histograma com (cv2.equalizeHist)
O problema 2 tem por missão implementar um modelo classificador para identificar se uma imagem é de um gato ou cachorro.

2 - Justificativa das Técnicas Utilizadas
R:
Problema 1:
Redimensionei as imagens para 128x128. Usei filtro gaussiano para reduzir ruídos e equalização de histograma para melhorar o contraste em imagens mais escuras. 

Problema 2:
Normalizei os pixels para tentar acelerar o treinamento, mas o hardware não permitiu que fosse mais rápido. Separei 80% dos dados para treino e 20% para teste para garantir avaliação justa. Usei uma CNN por ser eficaz em imagens. Avaliei com precisão, recall e F1-score e testei o modelo com minhas imagens para verificar a generalização.

3 - Etapas Realizadas
R:
Problema 1:
1° Etapa: Coleta de imagens externas, de gatos e cachorros.

2° Etapa: Redimensionamento para 128x128.

3° Etapa: Aplicação do filtro Gaussiano.

4° Etapa: Equalização de histograma.
-------------------------------------------------------------------------------
Problema 2:
1° Etapa: Normalização dos testes e treinos.

2° Etapa: Treinamento do Modelo usando dataset vindo Tensorflow, usando apenas as imagens de cachorro e gato.

3° Etapa: Separação em conjunto de treino e teste.

4° Etapa: Avaliação do modelo com métricas de precisão, recall e F1-score.

5° Etapa: Classificação das imagens externas com visualização dos resultados.

4 - Resultados Obtidos
R:
Acurácia no conjunto de teste: 

Precision: 

Recall: 

F1-score: 

5 - Tempo Total Gasto
R:
Aproximadamente  horas (divida o tempo estimado em: coleta de imagens, pré-processamento, modelagem e testes).

6 - Dificuldades Encontradas
R:
Fazer o treinamento ser mais rápido, mas por hardware eu tive essa dificuldade.

Saber se as imagens que escolhi são realmente boas para a predição.
