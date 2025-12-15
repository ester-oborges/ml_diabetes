### --- CONTEXTO E OBJETIVO ---
Este projeto utiliza um conjunto de dados do **Instituto Nacional de Diabetes e Doenças Digestivas e Renais**, composto exclusivamente por **mulheres adultas** (≥ 21 anos) **de ascendência indígena Pima**.

O objetivo é construir um modelo de *Machine Learning* capaz de **predizer a presença de diabetes** a partir de variáveis clínicas, com foco em **minimizar falsos negativos**, uma vez que a não detecção da doença representa um risco clínico maior do que alarmes falsos.

Dessa forma, as métricas priorizadas ao longo do projeto foram **Recall**, **F1-score** e **AUC-ROC**.

### --- ESTRUTURA DO REPOSITÓRIO ---

```text
├── diabetes.csv        # Conjunto de dados
├── ml_diabetes.ipynb   # Análise, pré-processamento e modelagem
├── README.md           # Documentação do projeto
└── LICENSE
```

### --- DATASET E INSPEÇÃO INICIAL ---
O dataset possui **768 registros** e **9 variáveis**, incluindo medições clínicas e o desfecho binário (*Outcome*).

Durante a inspeção inicial, observou-se que:

* Não há valores nulos explícitos nem registros duplicados;
* Algumas variáveis clínicas (*Glucose, BloodPressure, SkinThickness, Insulin, BMI*) apresentam zeros fisiologicamente impossíveis, indicando valores ausentes mascarados;
* As distribuições gerais são coerentes para um conjunto de dados clínico real.

### --- TRATAMENTO DOS DADOS ---
As principais decisões de pré-processamento foram:
* Conversão de valores impossíveis (zero) para `NaN`;
* Imputação pela **mediana**, estratégia robusta a outliers e adequada ao tamanho do dataset;
* Ajuste de tipos de dados para otimizar memória;
* Manutenção de outliers clinicamente plausíveis, evitando remoção automática de casos potencialmente relevantes.

### --- ANÁLISE EXPLORATÓRIA DE DADOS (EDA) ---
A EDA indicou que:
* *Glucose* e *BMI* apresentam relação mais forte com o diagnóstico de diabetes;
* *Age* atua como variável contextual, associada ao risco, mas com baixa separação isolada;
* A maioria das variáveis possui correlação fraca entre si, sugerindo baixa multicolinearidade;
* Algumas variáveis (ex.: *Insulin*) apresentam alta assimetria e variabilidade, exigindo cuidado no pré-processamento.

Esses achados orientaram tanto a escolha dos modelos quanto as decisões de avaliação.

### --- MODELO BASELINE — REGRESSÃO LOGÍSTICA ---
A Regressão Logística foi utilizada como modelo baseline por sua **simplicidade, interpretabilidade e ampla adoção em contextos clínicos**.

Foi aplicado `class_weight='balanced'` para lidar com o desbalanceamento das classes e priorizar a redução de falsos negativos. As variáveis foram escalonadas devido à sensibilidade do modelo à escala.

**Comparação de limiar de decisão:**
| Limiar | Recall | F1-score | Falsos Negativos | Falsos Positivos |
| --- | --- | --- | --- | --- |
| 0.5 (padrão) | 0.704 | 0.650 | alto | baixo |
| 0.3 (ajustado) | 0.889 | 0.676 | baixo | maior |

A redução do limiar de decisão resultou em **aumento expressivo do Recall**, alinhando o modelo ao objetivo clínico do projeto.

As curvas ROC e Precision–Recall indicam **boa capacidade discriminativa (AUC ≈ 0.81)** e um trade-off consciente entre sensibilidade e precisão.

### --- MODELO NÃO-LINEAR — RANDOM FOREST ---
O Random Forest foi avaliado para capturar **relações não lineares e interações complexas** entre variáveis clínicas. Foram aplicados ajustes para reduzir overfitting e manter o foco em sensibilidade.

**Comparação de limiar de decisão:**
| Limiar | Recall | F1-score | Falsos Negativos | Falsos Positivos |
| --- | --- | --- | --- | --- |
| 0.5 (padrão) | 0.667 | 0.649 | alto | baixo |
| 0.3 (ajustado) | 0.870 | 0.671 | baixo | maior |

O ajuste do limiar novamente mostrou-se essencial para reduzir falsos negativos, com **melhora do F1-score**, indicando equilíbrio adequado entre sensibilidade e precisão.
O modelo apresentou **AUC ≈ 0.82**, confirmando boa separação entre as classes.

**Avaliação de importância das variáveis:**
A análise de importância das variáveis no Random Forest indica que *Glucose*, *BMI* e *Age* são os principais fatores preditivos do modelo, em consonância com o conhecimento clínico sobre diabetes. As importâncias refletem contribuição preditiva global e não devem ser interpretadas como relações causais. Os resultados reforçam a coerência clínica do modelo e sua capacidade de capturar padrões relevantes nos dados.

### --- AVALIAÇÃO DE OVERFITTING ---
O risco de overfitting foi avaliado por meio da comparação entre métricas de treino e teste e pela estabilidade dos resultados.

Os modelos apresentaram **desempenho consistente**, indicando boa capacidade de generalização dentro das limitações do dataset.

### --- OTIMIZAÇÃO ---
Não foi realizado ajuste extensivo de hiperparâmetros via *GridSearchCV* ou *RandomizedSearchCV*, considerando:
* o **tamanho limitado do dataset**;
* o risco de sobreajuste;
* a priorização de **estabilidade, interpretabilidade e alinhamento clínico**.

### --- VALIDAÇÃO E LIMITAÇÕES ---
Este projeto apresenta algumas limitações importantes:
* Dataset restrito a um grupo populacional específico;
* Amostra de tamanho reduzido;
* O modelo **não substitui diagnóstico médico**, atuando apenas como ferramenta de apoio.

Para uso real, seria indispensável **validação externa** em outras populações e contextos clínicos.

### --- CONCLUSÃO ---
Os modelos avaliados demonstraram **desempenho consistente**, especialmente após o ajuste do limiar de decisão, que se mostrou fundamental para alinhar o comportamento dos classificadores ao objetivo clínico de **minimizar falsos negativos**.

A Regressão Logística forneceu um baseline interpretável, enquanto o Random Forest capturou interações mais complexas entre variáveis. Ambos se mostraram **suficientes para os objetivos do projeto**, sem necessidade imediata de modelos adicionais.

Como próximos passos, destacam-se:
* ampliação do conjunto de dados;
* validação em diferentes populações;
* aprofundamento da interpretabilidade;
* e, em um cenário aplicado, integração e monitoramento contínuo em ambientes clínicos.
