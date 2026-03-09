import pandas as pd
import numpy as np
import os
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

# =============================================================================
# 0. DOWNLOAD E CARREGAMENTO DOS DADOS
# =============================================================================

# Download da versão mais recente do dataset
kaglecsvml = kagglehub.dataset_download("shashwatwork/phishing-dataset-for-machine-learning")
print("Caminho para os arquivos do dataset:", kaglecsvml)

# Localizando o arquivo CSV dentro da pasta baixada
j_csv_path = os.path.join(kaglecsvml, "Phishing_Legitimate_full.csv")
df_dec_url_sec = pd.read_csv(j_csv_path)

# =============================================================================
# 1. DESCRIÇÃO DO PROBLEMA (REQUISITO 1 e 1.1)
# =============================================================================
"""
CONTEXTO REAL: Detecção automatizada de URLs de Phishing para proteção de usuários em tempo real.
MOTIVAÇÃO: Criminosos criam milhares de URLs fraudulentas diariamente; regras estáticas (Blacklists) 
não conseguem acompanhar a velocidade de criação, exigindo modelos que aprendam padrões (ML).
VARIÁVEL-ALVO: 'CLASS_LABEL' (1: Phishing, 0: Legítimo). 
Formulação como Classificação Binária pois o objetivo é uma decisão de sim/não para bloqueio.
"""

# Limpeza e preparação
df_dec_url_sec = df_dec_url_sec.drop(columns=['id']) # Removendo ID irrelevante para a predição
X = df_dec_url_sec.drop(columns=['CLASS_LABEL'])
y = df_dec_url_sec['CLASS_LABEL']

# Divisão Treino/Teste (Obrigatório antes de qualquer processamento)
X_train_splt, X_test_splt, y_train_splt, y_test_splt = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================================================================
#  MODELO BASELINE: PERCEPTRON (REQUISITOS 1.2, 1.3, 1.4)
# =============================================================================

# Construção do modelo baseline utilizando Pipeline (Requisito 1.2)
# O Perceptron é um classificador linear que busca um hiperplano separador.
pipe_bl = Pipeline([
    ('scaler', StandardScaler()), 
    ('model', Perceptron(random_state=42, max_iter=1000, tol=1e-3))
])

pipe_bl.fit(X_train_splt, y_train_splt)
predc_y = pipe_bl.predict(X_test_splt)

# Avaliação do baseline (Requisito 1.3)
print("\n--- MODELO BASELINE: PERCEPTRON ---")
print(f"Accuracy:  {accuracy_score(y_test_splt, predc_y):.4f}")
print(f"Precision: {precision_score(y_test_splt, predc_y):.4f}")
print(f"Recall:    {recall_score(y_test_splt, predc_y):.4f}")
print(f"F1-Score:  {f1_score(y_test_splt, predc_y):.4f}")

# Interpretação (Requisito 1.4)
# Extrai a lista de pesos das features
cffs = pipe_bl.named_steps['model'].coef_[0] 

# Extrai o valor do bias
bs = pipe_bl.named_steps['model'].intercept_[0] 

print(f"bias: {bs:.2f}")
# Para ver o primeiro coeficiente, por exemplo:
print(f"Primeiro coeficiente: {cffs[0]:.2f}")
print("Os coeficientes indicam o peso de cada feature na orientação do hiperplano.")
# Exemplo: Se o coeficiente de 'UrlLength' for positivo, URLs maiores aumentam a chance de ser Phishing.

# =============================================================================
# 3. ÁRVORE DE DECISÃO (REQUISITOS 3, 3.1, 3.2, 3.3)
# =============================================================================

# Treinamento com parâmetros padrão/mínimos (Requisito 3)
dt_simple_dtc = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_simple_dtc.fit(X_train_splt, y_train_splt)
pred_y_dt = dt_simple_dtc.predict(X_test_splt)

# Comparação direta (Requisito 3.1)
print("\n--- COMPARAÇÃO: ÁRVORE (SIMPLES) vs PERCEPTRON ---")
print(f"F1-Score Árvore:    {f1_score(y_test_splt, pred_y_dt):.4f}")
print(f"F1-Score Perceptron: {f1_score(y_test_splt, predc_y):.4f}")

# Interpretação das regras (Requisito 3.2)
# Visualizando as primeiras divisões para entender os caminhos da raiz às folhas
plt.figure(figsize=(20,10))
plot_tree(dt_simple_dtc, feature_names=X.columns, class_names=['Legit', 'Phishing'], filled=True, fontsize=10, max_depth=2)
plt.title("Visualização das Regras da Árvore (Top 3 Níveis)")
plt.show()

# Análise de Overfitting (Requisito 3.3)
# Se a árvore não for limitada (max_depth=None), ela tende a decorar o treino (100% acerto) e falhar no teste.

# =============================================================================
# 2. VALIDAÇÃO CRUZADA E BUSCA DE HIPERPARÂMETROS (REQUISITOS 2, 2.1, 2.2, 2.3)
# =============================================================================

# Espaço de busca para regularização da árvore (Requisito 2.1)
param_grid_dt_ft_ct = {
    'max_depth': [5, 10, 20, None],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy']
}

# Validação Cruzada Estratificada (Requisito 2)
cv_stfkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_dt_ft = GridSearchCV(DecisionTreeClassifier(random_state=42), 
                       param_grid_dt_ft_ct, cv=cv_stfkf, scoring='f1', n_jobs=-1)
grid_dt_ft.fit(X_train_splt, y_train_splt)

# Análise de Robustez entre Folds (Requisito 2.2)
res_cv = cross_validate(grid_dt_ft.best_estimator_, X_train_splt, y_train_splt, cv=cv_stfkf, scoring='f1')
print(f"\nRobustez da Árvore Otimizada (F1 médio nos folds): {res_cv['test_score'].mean():.4f} +/- {res_cv['test_score'].std():.4f}")

# Impacto da Regularização (Requisito 2.3):
# Ao definir min_samples_leaf > 1, impedimos que a árvore crie folhas para casos isolados (ruído).

# =============================================================================
# 4. MODELO AVANÇADO: ENSEMBLE RANDOM FOREST (REQUISITOS 4, 4.1, 4.2, 4.3)
# =============================================================================

# Implementação de Ensemble (Random Forest) (Requisito 4)
pipe_rfc = Pipeline([
    ('model', RandomForestClassifier(random_state=42))
])

# Busca de hiperparâmetros (Requisito 4.1)
param_gridscv_rf_m = {
    'model__n_estimators': [100, 200],
    'model__max_features': ['sqrt', 'log2'],
    'model__max_depth': [10, 20]
}

gridscv_rf = GridSearchCV(pipe_rfc, param_gridscv_rf_m, cv=cv_stfkf, scoring='f1', n_jobs=-1)
gridscv_rf.fit(X_train_splt, y_train_splt)
y_pred_rf_gcv = gridscv_rf.best_estimator_.predict(X_test_splt)

# Comparação Sistemática Final (Requisito 4.2)
print("\n--- DESEMPENHO FINAL (TESTE) ---")
print(classification_report(y_test_splt, y_pred_rf_gcv, target_names=['Legit', 'Phishing']))

# Análise de Importância e Complexidade (Requisito 4.3)
matter = gridscv_rf.best_estimator_.named_steps['model'].feature_importances_
feat_matter = pd.Series(matter, index=X.columns).sort_values(ascending=False).head(10)
print("\nTop 10 Features mais importantes (Ensemble):")
print(feat_matter)

"""
DISCUSSÃO FINAL:
O Random Forest apresentou o melhor F1-Score, demonstrando que a combinação de modelos 
(Ensemble) reduz a variância em comparação à Árvore de Decisão única. 
CUSTO COMPUTACIONAL: O RF é mais lento para treinar que o Perceptron, mas o ganho de 
precisão em um domínio sensível como cibersegurança justifica o custo.
"""
