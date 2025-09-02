# 🔧 Sistema de Manutenção Preditiva para Máquinas Industriais  
**Projeto Final do Bootcamp CDIA**  

## 📋 Descrição do Projeto  
Este projeto implementa um sistema inteligente de **manutenção preditiva** utilizando dados de sensores IoT para prever diferentes tipos de falhas em máquinas industriais.  

A solução envolve:  
- **Análise Exploratória de Dados (EDA)**  
- **Pré-processamento** (limpeza, imputação, normalização e codificação)  
- **Modelagem de Machine Learning** com múltiplos algoritmos  
- **Avaliação comparativa de desempenho**  

---

## 🎯 Objetivos  
- Identificar padrões em sensores industriais que indicam possíveis falhas.  
- Prever **falhas múltiplas** de forma simultânea.  
- Reduzir **tempo de inatividade** e **custos de manutenção**.  

---

## 📊 Dataset  
- **Fonte:** `bootcamp_train.csv`  
- **Registros:** 35.260 amostras  
- **Features:** 15 variáveis (sensores, parâmetros operacionais e falhas)  
- **Targets (falhas a serem previstas):**  
  - `FDF` – Falha por Desgaste da Ferramenta  
  - `FDC` – Falha por Dissipação de Calor  
  - `FP` – Falha por Potência  
  - `FTE` – Falha por Tensão Excessiva  
  - `FA` – Falha Aleatória  

### Exemplo de Features  
- `temperatura_ar`  
- `temperatura_processo`  
- `umidade_relativa`  
- `velocidade_rotacional`  
- `torque`  
- `desgaste_da_ferramenta`  

---

## 🛠️ Tecnologias Utilizadas  
- **Python 3.x**  
- **Bibliotecas:**  
  - `pandas`, `numpy` → manipulação de dados  
  - `matplotlib`, `seaborn` → visualização  
  - `scikit-learn` → modelagem e métricas  
  - `xgboost`, `lightgbm`, `catboost` → algoritmos de boosting  
  - `imblearn` → técnicas de balanceamento (SMOTE, undersampling)  

---

## 🔍 Etapas do Projeto  
1. **EDA**  
   - Verificação de nulos e inconsistências.  
   - Distribuição das variáveis e correlação.  
   - Análise do desbalanceamento das classes.  

2. **Pré-Processamento**  
   - Limpeza de inconsistências (`sim`, `não`, `0`, `1`, `y` etc.).  
   - Imputação de valores ausentes (mediana/moda).  
   - Normalização com **StandardScaler**.  
   - Codificação da variável categórica `tipo`.  

3. **Divisão dos Dados**  
   - 80% treino | 20% teste (estratificado).  

4. **Modelagem**  
   - Algoritmos avaliados:  
     - 🌲 Random Forest  
     - 🌐 Gradient Boosting  
     - ⚡ XGBoost  
     - 🔥 LightGBM  
     - 🐱 CatBoost  
   - Uso de **MultiOutputClassifier** para prever falhas múltiplas.  

5. **Avaliação**  
   - Métricas: Acurácia, Precisão, Recall, F1-Score e ROC AUC.  
   - Avaliação **por classe** e **média ponderada**.  

---

## 📊 Análise Exploratória dos Dados  
- **Valores nulos:** presentes em variáveis como `temperatura_ar`, `torque`, `desgaste_da_ferramenta`.  
- **Classes desbalanceadas:** algumas falhas representam **menos de 1% dos dados**.  
- **Outliers:** encontrados em temperatura, velocidade rotacional e torque.  

### Visualizações  
- Distribuição das Features Numéricas  
<img width="1494" height="990" alt="Distribuição das Variáveis" src="https://github.com/user-attachments/assets/106915fd-a516-481a-9cf9-a4461d59cf4e" />  

- Matriz de Correlação  
<img width="1366" height="663" alt="Matriz de Correlação" src="https://github.com/user-attachments/assets/bf98be17-69a9-4e4d-90be-2ac3dfdf50a7" />  

- Desbalanceamento das Classes  
<img width="1366" height="663" alt="Desbalanceamento" src="https://github.com/user-attachments/assets/8f5c823e-a2b7-48ac-820a-78b70a903dad" />  

**Percentual de falhas positivas:**  
- FDF: 0.20%  
- FDC: 0.63%  
- FP: 0.36%  
- FTE: 0.48%  
- FA: 0.21%  

👉 **Conclusão inicial:** dataset altamente desbalanceado → modelos tendem a alta acurácia e baixo recall.  

---

## 🤖 Resultados Obtidos  

### 🔍 Desempenho Médio por Modelo  
| Modelo            | Acurácia | Precisão | Recall | F1-Score | ROC AUC |
|------------------|----------|----------|--------|----------|---------|
| Random Forest     | 0.9966   | 0.4305   | 0.1671 | 0.2318   | 0.5831  |
| Gradient Boosting | 0.9963   | 0.3810   | 0.1785 | 0.2423   | 0.5882  |
| XGBoost           | 0.9966   | 0.5097   | 0.1968 | 0.2722   | 0.5975  |
| LightGBM          | 0.9963   | 0.4476   | 0.1802 | 0.2502   | 0.5897  |
| CatBoost          | 0.9964   | 0.4643   | 0.1833 | 0.2562   | 0.5913  |

### 📉 Detalhe do Melhor Modelo (XGBoost)  
| Falha | Acurácia | Precisão | Recall | F1-Score | ROC AUC |
|-------|----------|----------|--------|----------|---------|
| FDF   | 0.998    | 0.500    | 0.014  | 0.027    | 0.507   |
| FDC   | 0.994    | 0.571    | 0.202  | 0.299    | 0.601   |
| FP    | 0.997    | 0.667    | 0.044  | 0.083    | 0.522   |
| FTE   | 0.996    | 0.400    | 0.024  | 0.045    | 0.512   |
| FA    | 0.998    | 0.500    | 0.005  | 0.010    | 0.502   |

> 🔎 Apesar da alta acurácia, o recall é baixo para classes minoritárias → indicando que falhas raras ainda não são bem detectadas.  

---

## 🚀 Possíveis Melhorias  
- Balanceamento avançado: **SMOTE + undersampling**.  
- Otimização com **GridSearchCV** ou **Optuna**.  
- Testar **Deep Learning (LSTM, Autoencoders)** para séries temporais.  
- Pipeline automatizado para **deploy em produção** (Flask/FastAPI).  

---

## 📌 Conclusões  
- O projeto mostrou que, embora a acurácia seja altíssima (>99%), isso se deve ao **forte desbalanceamento**.  
- **XGBoost** foi o algoritmo mais equilibrado, mas ainda com recall baixo para falhas raras.  
- Futuras melhorias devem focar em **balanceamento das classes** e em **métricas robustas** além da acurácia.  

---

✍️ Autor: **Adilson**  
📅 Projeto Final – Bootcamp CDIA  
