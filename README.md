### --- CONTEXTO E OBJETIVO ---
Este conjunto de dados é originalmente do Instituto Nacional de Diabetes e Doenças Digestivas e Renais. Seu objetivo é prever, de forma diagnóstica, se uma paciente possui diabetes ou não, com base em determinadas medições clínicas.

Diversas restrições foram impostas na seleção das instâncias a partir de um banco de dados maior. Em particular, **todas as pacientes são mulheres, com pelo menos 21 anos de idade, de ascendência indígena Pima**.

O objetivo desse projeto é construir um modelo de Machine Learning capaz de prever se uma paciente tem diabetes ou não, com base nas variáveis clínicas disponíveis.

Neste contexto de saúde, onde a minimizaçāo de falsos negativos (Recall) é crítica — pois errar um diagnóstico positivo é mais grave —, as métricas de sucesso desde o início devem incluir o **Recall, o AUC-ROC e o F1-score** para garantir a performance e robustez do modelo.

### --- ESTRUTURA DO REPOSITÓRIO ---

```text
├── diabetes.csv        # Conjunto de dados utilizado no projeto
├── ml_diabetes.ipynb   # Notebook com análise, pré-processamento e modelagem
├── README.md           # Documentação do projeto
└── LICENSE             # Licença de uso do código
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
* **Outliers:** A análise inicial de outliers, utilizando o critério de 1.5 × IQR, indicou baixa incidência de valores extremos na maioria das variáveis, reforçando a consistência geral do conjunto de dados. Destacam-se, entretanto, Insulin e DiabetesPedigreeFunction, que concentraram maior número de valores fora do intervalo interquartil, comportamento esperado dada a natureza biológica e a alta variabilidade inerente a essas medidas. Em contextos de saúde, tais outliers não devem ser interpretados automaticamente como erros, pois podem representar casos clínicos reais e relevantes para a modelagem.

### --- ANÁLISE EXPLORATÓRIA DE DADOS (EDA) ---
**Heatmap de correlação entre as variáveis**
* Outcome → Apresenta correlação positiva moderada com Glucose e BMI, indicando relação direta entre níveis glicêmicos, índice de massa corporal e o diagnóstico de diabetes.
* Age x Pregnancies → Observa-se correlação positiva entre as duas variáveis, comportamento esperado do ponto de vista demográfico.
* Visão geral → A maioria das variáveis apresenta correlações fracas entre si, sugerindo baixa multicolinearidade e indicando que o modelo deverá capturar padrões principalmente por combinações não lineares.

**Histogramas (Outcome)**
* Glucose → Apresenta a separação mais evidente entre os grupos, com maior concentração de valores elevados no grupo com diabetes, indicando forte capacidade de distinção já na análise univariada.
* BMI → Exibe leve deslocamento da distribuição para valores mais altos no grupo com diabetes, embora com sobreposição considerável entre as classes, sugerindo influência moderada no diagnóstico.
* Age → Mostra deslocamento da distribuição para idades mais elevadas no grupo com diabetes, com sobreposição moderada, indicando que a variável contribui para o risco, mas não separa claramente os grupos isoladamente.
* Insulin → A análise por histogramas evidencia distribuição fortemente assimétrica, alta variabilidade e ampla sobreposição entre as classes, sugerindo baixo poder discriminativo isolado e potencial introdução de ruído sem tratamento prévio.
* BloodPressure → Apresenta distribuições muito semelhantes entre os grupos, com forte sobreposição, indicando baixo poder discriminativo quando analisada de forma isolada.

**Boxplots (Outcome)**
* Glucose → Apresenta assimetria positiva em ambos os grupos, com maior variabilidade entre indivíduos com diabetes (AIQ mais alto) e boa separação entre as distribuições, indicando forte capacidade discriminativa. Em machine learning, é uma variável central e altamente preditiva, tanto isoladamente quanto em interação com outras variáveis metabólicas.
* BMI → Mostra assimetria positiva e ampla sobreposição entre os grupos, com dispersão ligeiramente maior no grupo sem diabetes e outliers altos mais frequentes entre diabéticos. Isoladamente possui baixo poder discriminativo, mas torna-se relevante em interações com variáveis como Glucose e Age para identificar perfis de risco.
* Age → Exibe assimetria positiva, com concentração em idades menores e distribuições semelhantes entre os grupos, resultando em fraca separação direta. Em modelos de ML, atua melhor como variável moderadora ao contextualizar o risco metabólico ao longo do ciclo de vida.
* Insulin → A inspeção via boxplots aprofunda a análise ao revelar assimetria positiva extrema, grande concentração de outliers elevados e dispersão central reduzida, especialmente entre indivíduos com diabetes. Esses padrões indicam a necessidade de pré-processamento — como transformações ou tratamento de valores extremos — para que a variável possa contribuir de forma adequada em modelos de machine learning.
* BloodPressure → Possui distribuição aproximadamente simétrica, variabilidade semelhante entre os grupos e poucos outliers, resultando em baixa distinção entre diabéticos e não diabéticos. Em ML, tende a apresentar baixo poder preditivo isolado, sendo mais adequada como variável complementar.

### --- DESEMPENHO: MODELO BASELINE (REGRESSÃO LOGÍSTICA COM FOCO EM RECALL) ---
A Regressão Logística foi utilizada como modelo baseline por sua simplicidade, interpretabilidade e ampla adoção em problemas de classificação binária na área da saúde. O uso de `class_weight='balanced'` foi adotado para lidar com o desbalanceamento das classes e priorizar a minimização de falsos negativos, aspecto crítico em contextos clínicos. O escalonamento das variáveis foi aplicado devido à sensibilidade do modelo à escala dos dados, garantindo estabilidade numérica e melhor convergência. O parâmetro `max_iter=1000` assegura a convergência do algoritmo, enquanto `random_state=42` garante reprodutibilidade dos resultados.


| Limiar | Recall | F1-score | Falsos Negativos | Falsos Positivos |
| --- | --- | --- | --- | --- |
| 0.5 (padrão) | 0.704 | 0.650 | alto | baixo |
| 0.3 (ajustado) | 0.889 | 0.676 | baixo | maior |

A redução do limiar de decisão resultou em diminuição significativa de falsos negativos, alinhando o comportamento do modelo ao objetivo clínico do projeto.





### --- DESEMPENHO: MODELO NÃO LINEAR (RANDOM FOREST) ---
O Random Forest foi escolhido como modelo não linear para capturar relações mais complexas e interações entre as variáveis clínicas, que não são plenamente modeladas por abordagens lineares. O ajuste de `class_weight='balanced'` mantém o foco na redução de falsos negativos, alinhado ao objetivo clínico do projeto. O parâmetro `min_samples_leaf=5` foi definido para reduzir overfitting e aumentar a capacidade de generalização do modelo, enquanto `n_estimators=300` proporciona maior estabilidade das previsões. O uso de `n_jobs=-1` otimiza o tempo de treinamento e `random_state=42` assegura reprodutibilidade.

**Métricas (Limiar de 0.5)**
* Recall: 0.6666666666666666
* F1: 0.6486486486486487
* AUC: 0.8161111111111111
                       

### --- CONCLUSÃO ---









### --- PRÉ-PROCESSAMENTO ---

THRESHOLD PADRÃO (0.5)
* Recall 70% → o modelo ainda deixa passar ~30% dos diabéticos (alto para contexto clínico).
* AUC 0.81 → excelente capacidade discriminativa para um modelo linear simples.
* F1 0.65 → bom equilíbrio, mas não é a métrica prioritária aqui.

O modelo sabe separar, mas o threshold padrão não é adequado ao custo clínico.

THRESHOLD AJUSTADO (~0.3)
* Recall ~89% → você reduz drasticamente falsos negativos.
* F1 aumentou, não caiu → ótimo sinal.
* Você provavelmente aceitou mais falsos positivos (esperado).

Isso é EXATAMENTE o comportamento desejado em saúde.

MATRIZ DE CONFUSÃO

|  | Predito Não | Predito Sim |
| --- | --- | --- |
| Real Não | TN = 60 | FP = 40 |
| Real Sim | FN = 6 | TP = 48 |
* Apenas 6 falsos negativos
* Recall = 48 / (48 + 6) ≈ 0.89

Resultado excelente do ponto de vista clínico.

* 40 pacientes sem diabetes seriam sinalizadas como risco
* Em contexto clínico: exames adicionais, monitoramento, mudança de estilo de vida.

Custo aceitável frente ao risco de perder um diagnóstico real.

O modelo privilegia sensibilidade (Recall) de forma consciente, reduzindo drasticamente falsos negativos ao custo de mais falsos positivos — comportamento desejável em triagem clínica.

CURVA ROC (AUC = 0.81)
* O modelo separa bem as classes
* A curva se mantém claramente acima da diagonal
* AUC > 0.8 → boa capacidade discriminativa

Independentemente do threshold, o modelo tem boa habilidade em ranquear pacientes por risco de diabetes.

CURVA PRECISION-RECALL (AP = 0.67)
* Para Recall entre 0.8 e 0.9, a Precisão fica ~0.6. Ou seja: A cada 10 pacientes sinalizadas como diabéticas, ~6 realmente são. E isso é totalmente aceitável, especialmente se o exame confirmatório for barato (ex: glicemia).

O modelo de regressão logística apresentou AUC-ROC de 0.81, indicando boa capacidade discriminativa. Ao ajustar o threshold de decisão para priorizar sensibilidade, o Recall atingiu aproximadamente 89%, reduzindo significativamente o número de falsos negativos (6 casos). Embora isso tenha aumentado o número de falsos positivos, tal trade-off é considerado aceitável em contextos de triagem clínica, onde o custo de um falso negativo é substancialmente maior do que o de um falso positivo.






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
