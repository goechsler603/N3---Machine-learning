<div align="center">

# Previsão de Sucesso de Vendas de Videogames

**Projeto N3 de Aprendizagem de Máquina**

Dados a plataforma, o gênero, o publisher e o ano de lançamento de um jogo, o modelo prevê se ele será um **sucesso de vendas**.

</div>

---

## Sumário

- [Objetivo](#objetivo)
- [Dados](#dados)
- [Como executar](#como-executar)
- [Demonstração de uso](#demonstração-de-uso)
- [Metodologia](#metodologia)
- [Resultados](#resultados)
- [Limitações e próximos passos](#limitações-e-próximos-passos)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Integrantes](#integrantes)

---

## Objetivo

Prever se um jogo terá **vendas globais acima do percentil 75** do dataset, ou seja, **mais de 0,47 milhão de unidades**.

| | |
|---|---|
| **Tipo de problema** | Classificação binária (sucesso / não-sucesso) |
| **Comparação** | Regressão que prevê o valor de `Global_Sales` |
| **Quem poderia usar** | Publishers e estúdios que querem avaliar o potencial comercial de um lançamento antes de investir nele |

---

## Dados

| | |
|---|---|
| **Fonte** | [Kaggle: Video Games Sales](https://www.kaggle.com/competitions/video-games-sales/overview) |
| **Arquivo** | `Video_Games_Sales_as_at_22_Dec_2016.csv` |
| **Tamanho** | 16.719 jogos × 16 colunas (16.448 depois da limpeza) |
| **Features** | `Platform`, `Genre`, `Publisher`, `Year_of_Release` |
| **Alvo** | `Hit` = 1 se `Global_Sales` ≥ 0,47 mi (25% dos jogos) e 0 caso contrário |

---

## Como executar

### Opção 1: Kaggle (recomendado)

1. Abra o notebook `videogame.ipynb` no Kaggle.
2. Em **Add Input**, adicione o dataset da competição *Video Games Sales*.
3. Clique em **Run All**.

### Opção 2: Localmente

```bash
git clone <URL-DO-REPOSITORIO>
cd <NOME-DO-REPOSITORIO>
pip install -r requirements.txt
```

1. Baixe o CSV no Kaggle e salve em `data/Video_Games_Sales_as_at_22_Dec_2016.csv`.
2. No notebook, troque a variável `DATA_PATH` na célula de configuração:
   ```python
   DATA_PATH = "data/Video_Games_Sales_as_at_22_Dec_2016.csv"
   ```
3. Execute todas as células:
   ```bash
   jupyter notebook videogame.ipynb
   ```

---

## Demonstração de uso

A função `prever()` recebe os dados de um jogo novo e devolve a chance de ele ser um sucesso:

```python
prever("PS4", "Action", "Activision", 2015)
```
```
Action | PS4 | Activision | 2015 -> 54% de chance de sucesso (ALTO potencial de sucesso)
```

No fim, o notebook salva o modelo em `modelo_vendas.joblib`. Para usá-lo fora do notebook:

```python
import joblib
import pandas as pd

modelo = joblib.load("modelo_vendas.joblib")
jogo = pd.DataFrame([{
    "Platform": "PS4",
    "Genre": "Action",
    "Publisher": "Activision",
    "Year_of_Release": 2015,
}])
print(f"Chance de sucesso: {modelo.predict_proba(jogo)[0, 1]:.0%}")
```

---

## Metodologia

```
Dados brutos → EDA → Limpeza → Pré-processamento → Modelos → Validação → Avaliação → Demonstração
```

| Etapa | O que foi feito |
|---|---|
| **1. Exploração (EDA)** | Distribuição das vendas (muito assimétrica), venda média por gênero e plataforma, vendas por ano, principais publishers |
| **2. Limpeza** | Remoção de duplicatas e das 271 linhas sem ano ou gênero; publisher ausente substituído por `"Unknown"` |
| **3. Pré-processamento** | One-hot encoding nas categóricas (publishers com menos de 10 jogos agrupados) e padronização do ano |
| **4. Modelos** | Baseline aleatório, Regressão Logística, Random Forest e Gradient Boosting, todos com `class_weight="balanced"` |
| **5. Validação** | Divisão treino/teste 80/20 estratificada, validação cruzada com 5 folds e ajuste de hiperparâmetros com `RandomizedSearchCV` |
| **6. Métrica principal** | **F1** da classe "sucesso" |

> **Por que F1, e não acurácia?** As classes são desbalanceadas (25% / 75%). Um modelo que sempre responde "não-sucesso" já teria 75% de acurácia sem aprender nada.

---

## Resultados

Resultados no conjunto de teste (3.290 jogos):

| Modelo | Accuracy | Precision | Recall | F1 | ROC AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Gradient Boosting** | 0,741 | 0,492 | **0,760** | **0,597** | **0,825** |
| Gradient Boosting (ajustado) | 0,743 | 0,494 | 0,748 | 0,595 | 0,822 |
| Random Forest | 0,741 | 0,491 | 0,724 | 0,585 | 0,818 |
| Regressão Logística | 0,725 | 0,471 | 0,745 | 0,577 | 0,810 |
| Random Forest (ajustado) | 0,731 | 0,477 | 0,713 | 0,572 | 0,806 |
| Baseline (aleatório) | 0,626 | 0,257 | 0,255 | 0,256 | 0,503 |

### Principais conclusões

- O **Gradient Boosting** teve **F1 = 0,60**, mais que o dobro do baseline (0,26).
- Ele encontra **76% dos sucessos**, mas só cerca de metade dos jogos que aponta como sucesso realmente vende bem. Funciona bem como **triagem**, não como garantia.
- **O que mais influencia** (importância por permutação): **Publisher** ≫ Platform > ano > gênero. Quem publica pesa mais do que o tipo de jogo.
- **Onde o modelo erra:**
  - gêneros com vendas muito desiguais: Shooter, Racing e Action;
  - sucessos inesperados que ele não previu: *Zumba Fitness*, *Minecraft*;
  - jogos de publishers grandes que venderam pouco, mas que ele previu como sucesso.
- **Regressão (comparação):** R² = 0,31 na escala log. Prever o número exato de vendas é bem mais difícil do que prever se o jogo será um sucesso.

---

## Limitações e próximos passos

| Limitação | Próximo passo |
|---|---|
| Só 4 variáveis: sem franquia, marketing ou nota da crítica | Incluir `Critic_Score`, `User_Score`, `Rating` e uma feature de franquia |
| A divisão treino/teste é aleatória | Usar uma divisão temporal (treinar no passado, testar no futuro) |
| O dataset vai até 2016 e não tem vendas digitais | Atualizar com dados mais recentes |
| O limiar de decisão é fixo em 0,5 | Ajustar o limiar conforme o uso (priorizar precision ou recall) |

---

## Estrutura do repositório

```
projeto-n3
 ┣ videogame.ipynb      # notebook completo: EDA → modelagem → avaliação → demonstração
 ┣ requirements.txt     # dependências
 ┣ README.md
 ┗ data/                # coloque aqui o CSV baixado do Kaggle
```

---

## Integrantes

| Integrante |
|---|---|
| **Guilherme Oechsler** 
| **Jeliel Flores Moura** 
| **José Mateus Coelho Loose**
