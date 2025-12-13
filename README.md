--- CONTEXTO --- <br>
Este conjunto de dados é originalmente do Instituto Nacional de Diabetes e Doenças Digestivas e Renais. Seu objetivo é prever, de forma diagnóstica, se uma paciente possui diabetes ou não, com base em determinadas medições clínicas.

Diversas restrições foram impostas na seleção das instâncias a partir de um banco de dados maior. Em particular, **todas as pacientes são mulheres, com pelo menos 21 anos de idade, de ascendência indígena Pima**.

--- OBJETIVO --- <br>
Construir um modelo de Machine Learning capaz de prever se uma paciente tem diabetes ou não, com base nas variáveis clínicas disponíveis.

--- VISÃO GERAL E INSPEÇÕES INICIAIS DO DATASET --- <br>
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

--- TRATAMENTO DOS DADOS --- <br>
* **Tipos:** As variáveis Glucose, BloodPressure, SkinThickness, Insulin e BMI com valores mínimos iguais à zero devem ser convertidas para float, para que passem a aceitar valores NaN durante o tratamento. A variável Outcome será transformada em boolean e as demais serão reduzidas em precisão como sugerido durante a inspeção inicial.
* **Valores impossíveis:** Considerando que foi analisado que o dataset não apresentou valores nulos a princípio, mas possui valores ausentes mascarados como zero, tais valores foram substituídos por NaN (valores nulos) durante o tratamento dos dados para não causar distorções.
* **Outliers:** A análise de outliers pelo critério de 1.5 * IQR indicou baixa incidência de valores extremos na maioria das variáveis, com maior dispersão observada em Insulin e DiabetesPedigreeFunction, comportamento esperado dada a natureza dessas medidas.
