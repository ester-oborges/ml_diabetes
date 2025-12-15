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

### --- DESEMPENHO: MODELO BASELINE (REGRESSÃO LOGÍSTICA) ---
A Regressão Logística foi utilizada como modelo baseline por sua simplicidade, interpretabilidade e ampla adoção em problemas de classificação binária na área da saúde. O uso de `class_weight='balanced'` foi adotado para lidar com o desbalanceamento das classes e priorizar a minimização de falsos negativos, aspecto crítico em contextos clínicos. O escalonamento das variáveis foi aplicado devido à sensibilidade do modelo à escala dos dados, garantindo estabilidade numérica e melhor convergência. O parâmetro `max_iter=1000` assegura a convergência do algoritmo, enquanto `random_state=42` garante reprodutibilidade dos resultados.

| Limiar | Recall | F1-score | Falsos Negativos | Falsos Positivos |
| --- | --- | --- | --- | --- |
| 0.5 (padrão) | 0.704 | 0.650 | alto | baixo |
| 0.3 (ajustado) | 0.889 | 0.676 | baixo | maior |

A redução do limiar de decisão resultou em diminuição significativa de falsos negativos, alinhando o comportamento do modelo ao objetivo clínico do projeto.

**Matriz de confusão, curva ROC e Precisão-Recall**
Os resultados indicam que o modelo apresenta boa capacidade discriminativa (AUC ≈ 0.81) e comportamento coerente com o objetivo clínico do projeto. A matriz de confusão evidencia alto Recall, com baixo número de falsos negativos, ao custo de um aumento controlado de falsos positivos. A curva Precision–Recall reforça o trade-off adotado, mostrando que a priorização da sensibilidade impacta a precisão, decisão justificada em um cenário onde a não detecção da doença é mais crítica do que alarmes falsos.

### --- DESEMPENHO: MODELO NÃO LINEAR (RANDOM FOREST) ---
O Random Forest foi escolhido como modelo não linear para capturar relações mais complexas e interações entre as variáveis clínicas, que não são plenamente modeladas por abordagens lineares. O ajuste de `class_weight='balanced'` mantém o foco na redução de falsos negativos, alinhado ao objetivo clínico do projeto. O parâmetro `min_samples_leaf=5` foi definido para reduzir overfitting e aumentar a capacidade de generalização do modelo, enquanto `n_estimators=300` proporciona maior estabilidade das previsões. O uso de `n_jobs=-1` otimiza o tempo de treinamento e `random_state=42` assegura reprodutibilidade.

| Limiar | Recall | F1-score | Falsos Negativos | Falsos Positivos |
| --- | --- | --- | --- | --- |
| 0.5 (padrão) | 0.667 | 0.649 | alto | baixo |
| 0.3 (ajustado) | 0.870 | 0.671 | baixo | maior |
                       
A redução do limiar de decisão no Random Forest resultou em aumento significativo do Recall, reduzindo falsos negativos — aspecto crítico em um contexto clínico. Observa-se um aumento esperado de falsos positivos, porém com melhora do F1-score, indicando um trade-off equilibrado entre sensibilidade e precisão.

**Matriz de confusão, curva ROC e Precisão-Recall**
O Random Forest apresentou boa capacidade discriminativa (AUC = 0.82) e manteve baixo número de falsos negativos, característica essencial em um contexto clínico. O ajuste do limiar favorece o Recall, reduzindo o risco de não identificação de pacientes diabéticos, ainda que com aumento controlado de falsos positivos.

### --- RISCO DE OVERFITTING ---
O risco de overfitting foi avaliado por meio da comparação entre métricas de treino e teste, bem como pela estabilidade dos resultados em validação cruzada. O modelo apresentou desempenho consistente entre os conjuntos, indicando boa capacidade de generalização.

### --- OTIMIZAÇÃO ---
Não foi aplicado ajuste extensivo de hiperparâmetros via GridSearchCV ou RandomizedSearchCV, considerando o tamanho reduzido do dataset e o risco de overfitting. A abordagem adotada priorizou estabilidade, interpretabilidade e alinhamento clínico.

### --- VALIDAÇÃO E LIMITAÇÕES ---
Este projeto foi desenvolvido com um conjunto de dados restrito a um grupo populacional específico (mulheres adultas de ascendência indígena Pima), o que limita a generalização dos resultados. Além disso, o tamanho reduzido da amostra impõe restrições à complexidade dos modelos e aumenta o risco de overfitting. O modelo proposto não substitui diagnóstico médico, devendo ser interpretado como ferramenta de apoio à decisão. Para uso real, seria indispensável validação externa em outras populações e contextos clínicos.

### --- CONCLUSÃO ---
Os modelos avaliados apresentaram desempenho consistente, com destaque para o aumento significativo de Recall após o ajuste do limiar de decisão, alinhando o comportamento do classificador ao objetivo clínico de minimizar falsos negativos. Em cenários de triagem e apoio ao diagnóstico, o modelo pode auxiliar na identificação precoce de pacientes com maior risco de diabetes. Como próximos passos, destacam-se a ampliação do conjunto de dados, validações em diferentes populações, aprimoramento da interpretabilidade e, em um cenário aplicado, a integração e monitoramento contínuo do modelo em ambientes clínicos.

Foram avaliados modelos lineares e não lineares. A Regressão Logística foi utilizada como baseline interpretável, enquanto o Random Forest permitiu capturar interações mais complexas entre variáveis clínicas. Ambos apresentaram desempenho competitivo após ajuste de limiar de decisão, sendo considerados suficientes para os objetivos do projeto. Testes com modelos adicionais poderiam ser realizados em trabalhos futuros, caso haja necessidade de ganhos marginais de performance.
