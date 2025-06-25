# metricas-avaliacao-aprendizado
Cálculo de Métricas de Avaliação de Aprendizado   Neste projeto, vamos calcular as principais métricas para avaliação de modelos de classificação de dados, como acurácia, sensibilidade (recall), especificidade, precisão e F-score.

# metricas-avaliacao-aprendizado

**Descrição:** Este projeto demonstra o cálculo de métricas de avaliação de aprendizado de máquina, incluindo a matriz de confusão, acurácia, precisão, recall, F1-score e a curva ROC (com AUC).

**Objetivo:** O objetivo principal é fornecer um exemplo claro e prático de como avaliar o desempenho de modelos de classificação usando scikit-learn (Python).

## Conteúdo

*   `main.ipynb`: Um notebook Jupyter Colab contendo o código principal para calcular as métricas e gerar a curva ROC.
*   `README.md`: Este arquivo, fornecendo informações sobre o projeto.

## Como Usar

1.  **Abrir no Google Colab:** Clique no seguinte link para abrir o `main.ipynb` diretamente no Google Colab:

    [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<SEU_USERNAME>/<SEU_REPOSITORIO>/blob/main/main.ipynb)
    
2.  **Executar as Células:** No Colab, execute as células de código sequencialmente.

3.  **Explorar o Código:**
    *   As primeiras células definem dados de exemplo (ou você pode substituir pelos seus próprios dados).
    *   O projeto demonstra o uso de um modelo de classificação do scikit-learn (por exemplo, `LogisticRegression`).
    *   A matriz de confusão é calculada e exibida usando seaborn.
    *   As métricas de avaliação (acurácia, precisão, recall, F1-score) são calculadas a partir da matriz de confusão.
    *   A curva ROC e a AUC são calculadas e plotadas usando a biblioteca `matplotlib`.

## Requisitos

*   Python 3
*   As seguintes bibliotecas Python (geralmente já instaladas no Google Colab):
    *   NumPy
    *   scikit-learn
    *   Matplotlib
    *   Seaborn

    Se precisar instalar, execute:
    ```bash
    pip install numpy scikit-learn matplotlib seaborn
    ```

## Código de Exemplo (Trecho)

**Calculando a Matriz de Confusão:**

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# y_true: Rótulos verdadeiros
# y_pred: Rótulos previstos pelo modelo

matriz_confusao = confusion_matrix(y_true, y_pred)

sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Rótulos Previstos')
plt.ylabel('Rótulos Verdadeiros')
plt.title('Matriz de Confusão')
plt.show()

Calculando a Curva ROC:

from sklearn import metrics
import matplotlib.pyplot as plt

# y_true: Rótulos verdadeiros
# y_scores: Probabilidades previstas pelo modelo (predict_proba)

fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
roc_auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend()
plt.show()

Notas Adicionais

- Substitua os dados de exemplo: Adapte o código para usar seus próprios dados e modelos de aprendizado de máquina.
- Interpretação: As métricas de avaliação fornecem informações importantes sobre o desempenho do seu modelo. Analise a matriz de confusão para entender os tipos de erros que o modelo está cometendo e interprete a curva ROC e a AUC para avaliar o desempenho geral do modelo em diferentes limiares de decisão.
- Escolha do Modelo: Experimente com diferentes modelos de classificação do scikit-learn para comparar o desempenho.

Licença

MIT License

Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.
