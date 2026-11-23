# N3---Machine-learning

Previsão de sucesso de vendas de videogames 🎮

Projeto N3 de Aprendizagem de Máquina. O modelo recebe a plataforma, o gênero, o publisher e o ano de lançamento de um jogo e prevê se ele será um sucesso de vendas.

Objetivo

Prever se um jogo vai ter vendas globais acima do percentil 75 do dataset, ou seja, mais de 0,47 milhão de unidades. Isso é uma classificação binária: sucesso ou não-sucesso. Para comparação, também treinamos uma regressão que prevê o valor de vendas globais.

Quem poderia usar: publishers e estúdios que querem avaliar o potencial comercial de um lançamento antes de investir nele, por exemplo para decidir em qual plataforma lançar ou como priorizar projetos.

Dados
Fonte: Kaggle, Video Games Sales. Arquivo Video_Games_Sales_as_at_22_Dec_2016.csv.
Tamanho: 16.719 jogos e 16 colunas. Depois da limpeza ficam 16.448 jogos.
Features usadas: Platform, Genre, Publisher e Year_of_Release.
Alvo: Hit = 1 se Global_Sales ≥ 0,47 milhão (25% dos jogos) e 0 caso contrário.
Como executar
Opção 1: Kaggle (recomendado)
Abra o notebook videogame.ipynb no Kaggle.
Adicione o dataset da competição Video Games Sales em Add Input.
Clique em Run All.
Opção 2: Localmente
bash
git clone <URL-DO-REPOSITORIO>
cd <NOME-DO-REPOSITORIO>
pip install -r requirements.txt
Baixe o CSV no Kaggle e salve como data/Video_Games_Sales_as_at_22_Dec_2016.csv.
No notebook, troque a variável DATA_PATH (célula de configuração) para:
python
DATA_PATH = "data/Video_Games_Sales_as_at_22_Dec_2016.csv"
Execute todas as células:
bash
jupyter notebook videogame.ipynb
Fazendo uma previsão

No fim, o notebook salva o modelo em modelo_vendas.joblib. A função prever() recebe um jogo novo e devolve a chance de ele ser um sucesso:

python
prever("PS4", "Action", "Activision", 2015)
# Action | PS4 | Activision | 2015 -> 54% de chance de sucesso (ALTO potencial de sucesso)

Para usar o modelo salvo fora do notebook:

python
import joblib, pandas as pd
modelo = joblib.load("modelo_vendas.joblib")
jogo = pd.DataFrame([{"Platform": "PS4", "Genre": "Action", "Publisher": "Activision", "Year_of_Release": 2015}])
print(modelo.predict_proba(jogo)[0, 1])
Metodologia
Exploração (EDA): distribuição das vendas (muito assimétrica), venda média por gênero e por plataforma, vendas por ano e principais publishers.
Limpeza:
remoção de duplicatas;
remoção das 271 linhas sem ano ou gênero;
publisher ausente substituído por "Unknown".
Pré-processamento:
one-hot encoding nas variáveis categóricas, com publishers raros (menos de 10 jogos) agrupados;
padronização do ano.
Modelos: baseline aleatório estratificado, Regressão Logística, Random Forest e Gradient Boosting. Todos usam class_weight="balanced" por causa do desbalanceamento das classes (25% / 75%).
Validação:
divisão treino/teste de 80/20, estratificada;
validação cruzada estratificada com 5 folds;
ajuste de hiperparâmetros com RandomizedSearchCV.
Métrica principal: F1 da classe "sucesso". A acurácia engana com classes desbalanceadas: um modelo que sempre responde "não-sucesso" acerta 75%.
Principais resultados

Resultados no conjunto de teste (3.290 jogos):

Modelo	Accuracy	Precision	Recall	F1	ROC AUC
Gradient Boosting	0,741	0,492	0,760	0,597	0,825
Gradient Boosting (ajustado)	0,743	0,494	0,748	0,595	0,822
Random Forest	0,741	0,491	0,724	0,585	0,818
Regressão Logística	0,725	0,471	0,745	0,577	0,810
Random Forest (ajustado)	0,731	0,477	0,713	0,572	0,806
Baseline (aleatório)	0,626	0,257	0,255	0,256	0,503
O Gradient Boosting foi o melhor modelo: F1 = 0,60 contra 0,26 do baseline. Ele encontra 76% dos sucessos, mas só cerca de metade dos jogos que ele aponta como sucesso realmente vende bem.
O que mais influencia (importância por permutação): Publisher ≫ Platform > ano > gênero. Quem publica pesa mais do que o tipo de jogo.
Onde o modelo erra:
gêneros com vendas muito desiguais (Shooter, Racing, Action);
sucessos inesperados que ele não previu (Zumba Fitness, Minecraft);
jogos de publishers grandes que venderam pouco, mas que ele previu como sucesso.
Regressão (comparação): R² = 0,31 na escala log. Prever o número exato de vendas é bem mais difícil do que prever se o jogo será um sucesso.
Limitações e próximos passos
O modelo usa só 4 variáveis. Ele não sabe se o jogo é de uma franquia, quanto foi investido em marketing nem qual foi a nota da crítica.
O dataset vai até 2016 e não inclui vendas digitais.
A divisão treino/teste é aleatória. Uma divisão temporal (treinar no passado e testar no futuro) seria mais realista.
Próximos passos:
incluir Critic_Score, User_Score e Rating;
criar uma feature de franquia;
usar uma divisão temporal;
ajustar o limiar de decisão conforme o uso.
Estrutura do repositório
├── videogame.ipynb        # notebook completo (EDA → modelagem → avaliação → demonstração)
├── requirements.txt       # dependências
├── README.md
└── data/                  # coloque aqui o CSV baixado do Kaggle
Integrantes
Integrante
Guilherme Oechsler
Jeliel Flores Moura
José Mateus Coelho Loose
