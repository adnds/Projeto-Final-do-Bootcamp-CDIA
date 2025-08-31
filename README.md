# Projeto-Final-do-Bootcamp-CDIA

# 🔧 Sistema de Manutenção Preditiva para Máquinas Industriais

## 📋 Descrição do Projeto
Este projeto implementa um sistema inteligente de **manutenção preditiva** utilizando dados de sensores IoT para prever diferentes tipos de falhas em máquinas industriais.  
A solução aplica técnicas de **pré-processamento, análise exploratória de dados (EDA)** e **modelagem de Machine Learning** com múltiplos algoritmos para classificar falhas específicas.

---

## 🎯 Objetivo
- Identificar padrões em sensores industriais que indicam possíveis falhas.  
- Prever falhas múltiplas em equipamentos de forma simultânea.  
- Reduzir tempo de inatividade e custos de manutenção com técnicas preditivas.  

---

## 📊 Dataset
- Fonte: `bootcamp_train.csv`
- Registros: **35.260 amostras**  
- Features: **15 colunas** (sensores, parâmetros operacionais e falhas).  
- Targets (falhas a serem previstas):
  - **FDF** – Falha por Desgaste da Ferramenta  
  - **FDC** – Falha por Dissipação de Calor  
  - **FP** – Falha por Potência  
  - **FTE** – Falha por Tensão Excessiva  
  - **FA** – Falha Aleatória  

### Exemplo de Features:
- `temperatura_ar`  
- `temperatura_processo`  
- `umidade_relativa`  
- `velocidade_rotacional`  
- `torque`  
- `desgaste_da_ferramenta`  

---

## 🛠️ Tecnologias Utilizadas
- **Python 3.x**  
- **Bibliotecas principais**:
  - `pandas`, `numpy` – manipulação de dados  
  - `matplotlib`, `seaborn` – visualização  
  - `scikit-learn` – modelagem e métricas  
  - `xgboost`, `lightgbm`, `catboost` – modelos de boosting  
  - `imblearn` – técnicas de balanceamento (SMOTE, undersampling)  

---

## 🔍 Etapas do Projeto
1. **Análise Exploratória de Dados (EDA)**  
   - Verificação de nulos, inconsistências e outliers.  
   - Distribuição e correlação entre variáveis.  
   - Detecção de desbalanceamento severo das classes.  

2. **Pré-Processamento**  
   - Limpeza de valores inconsistentes.  
   - Imputação de valores ausentes (mediana/moda).  
   - Codificação de variáveis categóricas.  
   - Normalização das features.  

3. **Divisão dos Dados**  
   - Treino (80%) e Teste (20%), com estratificação para manter distribuição.  

4. **Modelagem de Machine Learning**  
   - Modelos testados:
     - **Random Forest**  
     - **Gradient Boosting**  
     - **XGBoost**  
     - **LightGBM**  
     - **CatBoost**  
   - Utilização de **MultiOutputClassifier** para prever múltiplas falhas.  

5. **Avaliação**  
   - Métricas utilizadas:  
     - Acurácia  
     - Precisão  
     - Recall  
     - F1-Score  
     - ROC AUC  
   - Avaliação feita **por classe** e **média das métricas**.  

---

## 📈 Resultados
- Os modelos apresentaram **alta acurácia geral (>99%)**, mas devido ao **forte desbalanceamento** das classes de falhas, métricas como Recall e F1-Score foram baixas para falhas mais raras.  
- **XGBoost** e **LightGBM** mostraram desempenho mais robusto em Recall e ROC AUC comparado aos demais.  
- A detecção de falhas múltiplas foi rara (**0.05% dos casos**).  

---

## 🚀 Possíveis Melhorias
- Aplicar **técnicas avançadas de balanceamento** (SMOTE combinado com undersampling).  
- Realizar **tuning de hiperparâmetros** com GridSearch ou Optuna.  
- Testar arquiteturas de **Deep Learning (LSTM, Autoencoders)** para séries temporais.  
- Implementar um **pipeline automatizado** para deploy em ambiente de produção (ex.: Flask, FastAPI).  

---
## 📊 Resultados e Análises

### Matriz de Confusão
![Matriz de Confusão](img/matriz_confusao.png)

### Importância das Features
![Importância das Features](img/feature_importance.png)

### Curva ROC
![Curva ROC](img/curva_roc.png)
