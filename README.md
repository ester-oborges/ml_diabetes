### --- CONTEXTO E OBJETIVO ---
Este conjunto de dados é originalmente do Instituto Nacional de Diabetes e Doenças Digestivas e Renais. Seu objetivo é prever, de forma diagnóstica, se uma paciente possui diabetes ou não, com base em determinadas medições clínicas.

Diversas restrições foram impostas na seleção das instâncias a partir de um banco de dados maior. Em particular, **todas as pacientes são mulheres, com pelo menos 21 anos de idade, de ascendência indígena Pima**.

O objetivo desse projeto é construir um modelo de Machine Learning capaz de prever se uma paciente tem diabetes ou não, com base nas variáveis clínicas disponíveis.

Neste contexto de saúde, onde a minimizaçāo de falsos negativos (Recall) é crítica — pois errar um diagnóstico positivo é mais grave —, as métricas de sucesso desde o início devem incluir o **Recall, o AUC-ROC e o F1-score** para garantir a performance e robustez do modelo.

### --- ESTRUTURA DO REPOSITÓRIO ---

```text
├── diabetes.csv          # Conjunto de dados utilizado no projeto
├── ml_diabetes.ipynb     # Notebook com análise, pré-processamento e modelagem
├── README.md             # Documentação do projeto
└── LICENSE               # Licença de uso do código
```

### --- VISÃO GERAL E INSPEÇÕES INICIAIS DO DATASET ---
Foram identificadas 9 colunas no dataset, descritas a seguir:
* Pregnancies (gestações): Quantidade de vezes que a paciente esteve grávida.
* Glucose (glicose): Concentração plasmática de glicose (mg/dL).
* BloodPreassure (pressão arterial): Pressão arterial diastólica (mmHg).
* SkinThickness (espessura da pele): Espessura da dobra da pele do tríceps (mm).
* Insulin (insulina): Nível sérico de insulina (mu U/ml).
* BMI (IMC): Índice de Massa Corporal (kg/m²).
* DiabetesPedigreeFunction (função de herediariedade): Função que avalia o risco genético de diabetes com base no histórico familiar.
* Age (idade): Idade das pacientes em anos.
* Outcome (resultado): Diagnóstico de diabetes (0 = não, 1 = sim).

Também foram observadas as seguintes características:
* **Tamanho do dataset (`df.info()`):** Possui 768 registros. Para problemas de ML supervisionado, este é considerado um dataset de pequeno a médio porte, adequado para estudos exploratórios, prototipagem de modelos e validação de técnicas de pré-processamento.
* **Tipos dos dados (`df.info()`):** As variáveis numéricas estão originalmente classificadas como int64 e float64. É possível reduzir esses tipos para versões de menor precisão sem perda relevante de informação. Além disso, a variável Outcome pode ser convertida para tipo boolean, otimizando o uso de memória e o desempenho computacional.
* **Valores nulos (`df.info()`):** Não há valores nulos.
* **Duplicatas (`df.duplicated().sum()`):** Não há registros duplicados.
* **Resumo estatístico (`df.describe()`):** Foram detectados zeros fisiologicamente impossíveis nas variáveis Glucose, BloodPressure, SkinThickness, Insulin e BMI com valores mínimos iguais à zero. Ou seja, **o dataset não apresentou valores nulos a princípio, mas possui valores ausentes mascarados como zero**. As demais distribuições nesse resumo estatístico parecem coerentes.

### --- TRATAMENTO DOS DADOS ---
* **Tipos:** As variáveis Glucose, BloodPressure, SkinThickness, Insulin e BMI com valores mínimos iguais à zero devem ser convertidas para float, para que passem a aceitar valores NaN durante o tratamento. A variável Outcome será transformada em boolean e as demais serão reduzidas em precisão como sugerido durante a inspeção inicial.
* **Valores impossíveis:** Embora o dataset não apresente valores nulos explicitamente, foi identificado que algumas variáveis clínicas possuem valores ausentes mascarados como zero. Durante a etapa de pré-processamento, esses valores foram substituídos por NaN e posteriormente imputados utilizando a **mediana** de cada variável, estratégia escolhida por ser robusta à presença de outliers e adequada a conjuntos de dados clínicos de tamanho reduzido. Essa abordagem contribui para a preservação da distribuição original das variáveis, reduzindo o impacto de valores extremos e promovendo maior estabilidade ao modelo de machine learning.
* **Outliers:** A análise de outliers pelo critério de 1.5 * IQR indicou baixa incidência de valores extremos na maioria das variáveis, com maior dispersão observada em Insulin e DiabetesPedigreeFunction, comportamento esperado dada a natureza dessas medidas. Em saúde, outliers podem ser casos clínicos reais.

### --- ANÁLISE EXPLORATÓRIA DE DADOS (EDA) ---
**Heatmap de correlação entre as variáveis**
* Outcome → correlação positiva moderada com Glucose e BMI.
* Age x Pregnancies → correlação positiva entre si.
* Pouca correlação forte entre muitas variáveis.

**Histogramas (Outcome)**
* Glucose → Apresenta a separação mais clara entre os grupos, com valores mais elevados concentrados no grupo com diabetes.
* BMI → Exibe leve deslocamento para valores mais altos no grupo com diabetes, apesar de sobreposição considerável.
* Age → Mostra deslocamento para idades mais elevadas no grupo com diabetes, com sobreposição moderada entre as classes.
* Insulin → Possui distribuição assimétrica, alta variabilidade e grande sobreposição entre as classes, sendo uma variável ruidosa isoladamente.
* BloodPressure → Apresenta forte sobreposição entre os grupos, com distribuições semelhantes, indicando baixo poder discriminativo quando analisada isoladamente.

**Boxplots (Outcome)**
* Glucose → Apresenta assimetria positiva em ambos os grupos, com maior variabilidade entre indivíduos com diabetes (AIQ mais alto) e boa separação entre as distribuições, indicando forte capacidade discriminativa. Em machine learning, é uma variável central e altamente preditiva, tanto isoladamente quanto em interação com outras variáveis metabólicas.
* BMI → Mostra assimetria positiva e ampla sobreposição entre os grupos, com dispersão ligeiramente maior no grupo sem diabetes e outliers altos mais frequentes entre diabéticos. Para ML, tem baixo poder discriminativo isolado, mas ganha relevância quando combinada com variáveis como Glucose e Age para identificar perfis de risco.
* Age → Exibe assimetria positiva, com concentração em idades mais jovens e distribuições semelhantes entre os grupos, resultando em fraca separação direta. Em modelos de ML, atua melhor como variável moderadora, ajudando a capturar padrões de risco quando associada a indicadores metabólicos.
* Insulin → Apresenta assimetria positiva extrema, grande quantidade de outliers altos e dispersão central reduzida. Para ML, requer forte pré-processamento (transformações ou tratamento de outliers), podendo contribuir com informação relevante após ajuste adequado.
* BloodPressure → Possui distribuição aproximadamente simétrica e variabilidade semelhante entre os grupos, com poucos outliers e baixa distinção entre diabéticos e não diabéticos. Em ML, tende a ter baixo poder preditivo isolado, sendo mais útil como variável complementar em conjunto com outros fatores clínicos.





### --- PRÉ-PROCESSAMENTO ---
Separação de dados
```text
x = df.drop('Outcome', axis=1)
y = df['Outcome']
```
* `train_test_split`
* Estratificação por Outcome

### --- ESCALONAMENTO ---
Essencial para modelos baseados em distância:
* StandardScaler ou RobustScaler (usar pipeline para evitar data leakage).

### --- MODELAGEM ---
Modelos baseline:
* Logistic Regression (obrigatório!)
* KNN
* Decision Tree

Justificativa para README:
"Modelos simples fornecem interpretabilidade e referência de desempenho."

Modelos mais robustos:
* Random Forest
* Gradient Boosting
* XGBoost / LightGBM (opcional)

### --- AVALIAÇÃO DOS MODELOS ---
Métricas principais:
* Confusion Matrix
* Recall (classe positiva)
* Precision
* F1-score
* ROC-AUC

Análises importantes:
* Comparar desempenho vs. teste
* Avaliar overfitting
* Ajustar threeshold de decisão (opcional)

### --- OTIMIZAÇÃO ---
* GridSearchCV ou RandomizedSearchCV
* Foco em Recall e F1-score

Evitar over-otimização: explicar limites do dataset (tamanho pequeno).

### --- INTERPRETABILIDADE ---
Essencial em projetos de saúde.

Abordagens:
* Coeficientes (Logistic Regression)
* Feature Importance (árvores)
* SHAP (se quiser elevar o nível do projeto)

### --- VALIDAÇÃO E LIMITAÇÕES ---
Seção obrigatória para recrutadores.

Pontos a destacar:
* Dataset restrito a um grupo populacional.
* Tamanho limitado da amostra.
* Modelo não substitui diagnóstico médico.
* Necessidade de validação externa.

### --- CONCLUSÃO ---
Responder claramente:
* O modelo funciona?
* Em quais cenários seria útil?
* Quais próximos passos?

Próximos passos possíveis:
* Mais dados
* Validação cross-population
* Integração com sistemas clínicos
* Monitoraemtno em produção
