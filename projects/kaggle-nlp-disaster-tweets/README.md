# NLP Competition — Disaster Tweets

Notebook de portfólio para a competição Kaggle **Natural Language Processing with Disaster Tweets**. O objetivo é demonstrar um fluxo de Machine Learning reproduzível para classificação binária de textos.

## O que o notebook cobre

- Análise inicial dos dados e verificação de valores ausentes.
- Separação treino/validação estratificada e reproduzível.
- Baseline simples com DummyClassifier.
- Pipeline TF-IDF + Logistic Regression.
- Avaliação com F1, precisão, recall e matriz de confusão.
- Inspeção de falsos positivos e falsos negativos.
- Treinamento final com todos os dados rotulados e geração de submission.csv.

## Como executar no Kaggle

1. Entre na competição e aceite as regras: https://www.kaggle.com/competitions/nlp-getting-started
2. Crie um notebook associado à competição ou carregue/import a partir deste repositório.
3. Execute todas as células em ordem.
4. Confirme o F1 na validação e revise os erros antes de enviar submission.csv.
5. Publique o notebook executado no seu perfil Kaggle e adicione o link no seu GitHub.

O notebook espera que os arquivos estejam em /kaggle/input/nlp-getting-started/.

## Honestidade experimental

Este repositório fornece o código do experimento. Nenhum F1 ou posição de leaderboard é declarado aqui porque os resultados precisam ser obtidos ao executar o notebook no ambiente Kaggle. Registre a pontuação real, a configuração e as limitações depois de executar.

## Link do notebook

- notebook.ipynb: arquivo Jupyter para importação e execução no Kaggle.
