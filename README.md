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
## 📊 Análise Exploratória dos Dados

- **Tamanho do dataset**: 35.260 amostras e 15 colunas  
- **Valores nulos**: presentes em variáveis numéricas (`temperatura_ar`, `torque`, `desgaste_da_ferramenta`)  
- **Classes desbalanceadas**: algumas falhas representam **menos de 1% dos dados**  
- **Outliers** detectados em variáveis de temperatura, velocidade rotacional e torque  

### Distribuição das Features Numéricas
![Distribuição das Variáveis](imagens/distribuicao_variaveis.png)

### Correlação entre Variáveis
![Matriz de Correlação](imagens/matriz_correlacao.png)

### Desbalanceamento das Classes
![Desbalanceamento](imagens/desbalanceamento_classes.png)

- FDF: 0.20% positivos  
- FDC: 0.63% positivos  
- FP: 0.36% positivos  
- FTE: 0.48% positivos  
- FA: 0.21% positivos  

👉 **Conclusão inicial**: dataset altamente desbalanceado → risco de modelos com alta acurácia mas baixo recall para falhas.

---

## ⚙️ Pré-processamento

1. Limpeza de inconsistências (`sim`, `não`, `0`, `1`, `y`, etc.)  
2. Imputação de valores nulos → **mediana** para numéricos  
3. Normalização com **StandardScaler**  
4. Codificação da variável categórica `tipo`  
5. Divisão em **treino (80%)** e **teste (20%)**  

---

## 🤖 Modelos Avaliados

Foram avaliados os seguintes classificadores em **configuração MultiOutput**:

- 🌲 Random Forest  
- 🌐 Gradient Boosting  
- ⚡ XGBoost  
- 🔥 LightGBM  
- 🐱 CatBoost  

---

## 📈 Resultados

### Random Forest
- Acurácia Média: **0.9966**  
- Precisão Média: **0.4305**  
- Recall Médio: **0.1671**  
- F1-Score Médio: **0.2318**

### Gradient Boosting
- Acurácia Média: **0.9963**  
- Precisão Média: **0.3810**  
- Recall Médio: **0.1785**  
- F1-Score Médio: **0.2423**

### XGBoost
- Acurácia Média: **0.9966**  
- Precisão Média: **0.5097**  
- Recall Médio: **0.1968**  
- F1-Score Médio: **0.2722**

### LightGBM
- Acurácia Média: **0.9963**  
- Resultados similares ao XGBoost, mas com menor recall em algumas classes  

### CatBoost
- [Resultados ainda em execução ou a incluir aqui]  

---

## 📊 Comparação entre Modelos
| Modelo            | Acurácia Média | Precisão Média | Recall Médio | F1-Score Médio |
|-------------------|----------------|----------------|--------------|----------------|
| Random Forest     | 0.9966         | 0.4305         | 0.1671       | 0.2318         |
| Gradient Boosting | 0.9963         | 0.3810         | 0.1785       | 0.2423         |
| XGBoost           | 0.9966         | 0.5097         | 0.1968       | 0.2722         |
| LightGBM          | 0.9963         | ~0.45          | ~0.18        | ~0.25          |
| CatBoost          | —              | —              | —            | —              |

---

## 📌 Conclusões
- Apesar da **alta acurácia**, os modelos apresentaram **baixo recall e F1-score** devido ao **forte desbalanceamento das classes**.  
- **XGBoost** teve o melhor equilíbrio entre precisão e recall.  
- Recomenda-se aplicar técnicas de **oversampling (SMOTE)** ou **undersampling** para melhorar o desempenho nas classes minoritárias.  
- Futuras otimizações podem incluir **ajuste de hiperparâmetros via GridSearchCV** e uso de **métricas ponderadas**.

✍️ Autor: Adilson | Projeto Final Bootcamp CDIA
---
