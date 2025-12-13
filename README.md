--- CONTEXTO --- <br>
Este conjunto de dados é originalmente do Instituto Nacional de Diabetes e Doenças Digestivas e Renais. O objetivo do conjunto de dados é prever diagnosticamente se um paciente tem diabetes ou não, com base em certas medições diagnósticas incluídas no conjunto de dações. Diversas restrições foram impostas à seleção dessas instâncias de um banco de dados maior. Em particular, todas as pacientes aqui são mulheres com pelo menos 21 anos de ascendência indígena Pima.

--- CONTEÚDO --- <br>
Os conjuntos de dados consistem em várias variáveis preditoras médicas e uma variável-alvo. As variáveis preditoras incluem o número de gestações que a paciente teve, seu IMC, nível de insulina, idade e assim por diante.

--- OBJETIVO --- <br>
Construir um modelo Machine Learning para prever se os pacientes têm diabetes ou não.

--- INSPEÇÃO INICIAL --- <br>
Foi verificado (através de df.sample()) que a tabela possui 9 colunas, sendo elas:
* Pregnancies (gestações): quantidade de vezes que a paciente esteve grávida.
* Glucose (glicose): concentração plasmática de glicose (mg/dL).
* BloodPreassure (pressão arterial): pressão arterial diastólica (mmHg).
* SkinThickness (espessura da pele): espessura da dobra da pele do tríceps (mm).
* Insulin (insulina): nível sérico de insulina (mu U/ml).
* (IMC): Índice de Massa Corporal (kg/m²).
* DiabetesPedigreeFunction (função de herediariedade): função que avalia o risco genético de diabetes com base no histórico familiar.
* Age (idade): idade das pacientes em anos.
* Outcome (resultado): indica diagnóstico de diabetes (0 = não, 1 = sim).

Durante a inspeção inicial também foram observados os seguintes pontos:
* A princípio, não há valores nulos (df.info()), porém foi observado que há registros de valores iguais a zero ((df == 0).sum()) em colunas como blablablabla, indicando valores impossíveis -- portanto, esses valores foram substituídos por NaN durante o tratamento dos dados, já que (INSERIR UMA JUSTIFICATIVA BLABLA).
* O dataset possui 768 registros (df.info()), sendo este um dataset grande?pequeno?razoável? sei lá, chegar a alguma conclusão aqui.
* Os tipos numéricos foram classificados em float64 e int64 (df.info()), mas podem ser reduzidos de 64 bits para versões menores e a variável Outcome pode ser convertida para boolean, preservando a precisão necessária ao dataset e otimizando o uso de memória e o desempenho do processamento.

(FALTA ANALISAR O DESCRIBE)

--- ANÁLISE DE OUTLIERS --- <br>
O dicionário retornado indica quantos registros em cada coluna estão fora do intervalo esperado (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR).
* Pregnancies (4): algumas mulheres podem ter tido muitas gestações, provavelmente são dados legítimos.
* Glucose (5): podem haver algumas pacientes hiperglicêmicas, provavelmente são dados legítimos.
* **BloodPressure (45): muitos outliers, merece inspeção manual mais profunda.**
* SkinThickness (1): quase nenhuma ocorrência, provavelmente são dados legítimos.
* Insulin (34): geralmente a insulina é uma variável de distribuição assimétrica, outliers são esperados e provavelmente são dados legítimos.
* BMI (19): valores de IMC muito altos costumam ser genuínos, provavelmente são dados legítimos.
* DiabetesPedigreeFunction (29): geralmente a Pedigree Function é uma variável de distribuição assimétrica, outliers são esperados e provavelmente são dados legítimos.
* Age (9): algumas mulheres podem ter idades muito altas ou muito baixas, provavelmente são dados legítimos.
* Outcome (0): nenhum outlier, esperado pois é binária.

