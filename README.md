--- CONTEXTO --- <br>
Este conjunto de dados é originalmente do Instituto Nacional de Diabetes e Doenças Digestivas e Renais. Seu objetivo é prever, de forma diagnóstica, se uma paciente possui diabetes ou não, com base em determinadas medições clínicas.

Diversas restrições foram impostas na seleção das instâncias a partir de um banco de dados maior. Em particular, **todas as pacientes são mulheres, com pelo menos 21 anos de idade, de ascendência indígena Pima**.

--- OBJETIVO --- <br>
Construir um modelo de Machine Learning capaz de prever se uma paciente tem diabetes ou não, com base nas variáveis clínicas disponíveis.

--- INSPEÇÃO INICIAL --- <br>
A partir de uma inspeção inicial dos dados (utilizando `df.sample()`), foi identificado que o dataset possui 9 colunas, descritas a seguir:
* Pregnancies (gestações): Quantidade de vezes que a paciente esteve grávida.
* Glucose (glicose): Concentração plasmática de glicose (mg/dL).
* BloodPreassure (pressão arterial): Pressão arterial diastólica (mmHg).
* SkinThickness (espessura da pele): Espessura da dobra da pele do tríceps (mm).
* Insulin (insulina): Nível sérico de insulina (mu U/ml).
* BMI (IMC): Índice de Massa Corporal (kg/m²).
* DiabetesPedigreeFunction (função de herediariedade): Função que avalia o risco genético de diabetes com base no histórico familiar.
* Age (idade): Idade das pacientes em anos.
* Outcome (resultado): Diagnóstico de diabetes (0 = não, 1 = sim).

Durante a inspeção inicial, também foram observadas as seguintes características:
* **Tamanho do dataset (`df.info()`):** Possui 768 registros. Para problemas de ML supervisionado, este é considerado um dataset de pequeno a médio porte, adequado para estudos exploratórios, prototipagem de modelos e validação de técnicas de pré-processamento.
* **Tipos dos dados (`df.info()`):** As variáveis numéricas estão originalmente classificadas como int64 e float64. É possível reduzir esses tipos para versões de menor precisão sem perda relevante de informação. Além disso, a variável Outcome pode ser convertida para tipo boolean, otimizando o uso de memória e o desempenho computacional.
* **Duplicatas (`df.duplicated().sum()`):** Não há registros duplicados nesse dataset.
* **Valores nulos (`df.info()`):** A princípio, não há valores nulos explícitos.
* **Valores iguais a zero (`(df == 0).sum()`):** Foi identificado que algumas colunas apresentam valores iguais a zero em contextos biologicamente impossíveis (por exemplo: glicose, pressão arterial, espessura da pele, insulina e IMC). Esses valores não representam medições reais, mas registros ausentes codificados como zero. Portanto, tais valores foram substituídos por NaN (valores nulos) durante o tratamento dos dados para não causar distorções.

(FALTA ANALISAR O DESCRIBE)

--- ANÁLISE DE OUTLIERS --- <br>
O dicionário retornado indica quantos registros em cada coluna estão fora do intervalo esperado (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR).
* Pregnancies (4): Algumas mulheres podem ter tido muitas gestações, provavelmente são dados legítimos.
* Glucose (5): Podem haver algumas pacientes hiperglicêmicas, provavelmente são dados legítimos.
* **BloodPressure (45): Muitos outliers, merece inspeção manual mais profunda.**
* SkinThickness (1): Quase nenhuma ocorrência, provavelmente são dados legítimos.
* Insulin (34): Geralmente a insulina é uma variável de distribuição assimétrica, outliers são esperados e provavelmente são dados legítimos.
* BMI (19): Valores de IMC muito altos costumam ser genuínos, provavelmente são dados legítimos.
* DiabetesPedigreeFunction (29): Geralmente a Pedigree Function é uma variável de distribuição assimétrica, outliers são esperados e provavelmente são dados legítimos.
* Age (9): Algumas mulheres podem ter idades muito altas ou muito baixas, provavelmente são dados legítimos.
* Outcome (0): nenhum outlier, esperado pois é binária.

