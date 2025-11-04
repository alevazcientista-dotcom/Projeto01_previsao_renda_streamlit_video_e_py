import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Para treemap opcional
def try_import_squarify():
    try:
        import squarify  # type: ignore
        return squarify
    except Exception:
        return None

# Helpers
def read_info(df: pd.DataFrame) -> str:
    """Captura df.info() como texto para exibir no Streamlit."""
    buf = StringIO()
    df.info(buf=buf)
    return buf.getvalue()

def has_cols(df: pd.DataFrame, cols):
    return all(c in df.columns for c in cols)

@st.cache_data(show_spinner=False)
def load_csv_auto(path_candidate: str):
    if os.path.exists(path_candidate):
        return pd.read_csv(path_candidate)
    return None

def load_dataset():
    """Tenta carregar automaticamente; se não, oferece upload."""
    local_path = "previsao_de_renda.csv"  # mesmo diretório do app
    df = load_csv_auto(local_path)
    if df is not None:
        st.sidebar.success(f"Carregado automaticamente: {local_path}")
        return df

    st.sidebar.warning("Arquivo local não encontrado. Envie o CSV.")
    up = st.sidebar.file_uploader("Selecione o arquivo previsao_de_renda.csv", type=["csv"])
    if up:
        return pd.read_csv(up)
    return None

# Funções de pré-processamento e modelagem do segundo código
@st.cache_data
def preprocessar_dados(df):
    """Pré-processa os dados: trata valores ausentes e codifica variáveis categóricas."""
    if df is None:
        return None, None, None

    # Imputa valores ausentes em 'tempo_emprego' com a mediana
    if 'tempo_emprego' in df.columns and df['tempo_emprego'].isnull().any():
        mediana_tempo_emprego = df['tempo_emprego'].median()
        df['tempo_emprego'] = df['tempo_emprego'].fillna(mediana_tempo_emprego)

    # Identifica e codifica colunas categóricas
    colunas_categoricas = df.select_dtypes(include=['object', 'bool']).columns
    # Exclui 'data_ref' se não for uma feature relevante
    if 'data_ref' in colunas_categoricas:
        colunas_categoricas = colunas_categoricas.drop('data_ref')

    df_codificado = pd.get_dummies(df, columns=colunas_categoricas, drop_first=True)

    # Define matriz de features X e variável target y
    colunas_a_excluir = ['renda', 'Unnamed: 0', 'index', 'data_ref']
    X = df_codificado.drop([col for col in colunas_a_excluir if col in df_codificado.columns], axis=1)
    y = df_codificado['renda']

    return X, y, df_codificado.columns.tolist()  # Retorna colunas para manipulação de inputs

@st.cache_resource
def treinar_modelo(X, y):
    """Treina um modelo de Regressão Linear."""
    if X is None or y is None:
        return None, None, None, None

    # Divide os dados em treino e teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

    # Inicializa e treina o modelo
    modelo_regressao = LinearRegression()
    modelo_regressao.fit(X_treino, y_treino)

    return modelo_regressao, X_teste, y_teste, X_treino.columns.tolist()  # Retorna modelo e nomes de features

def fazer_previsoes(modelo, dados_entrada):
    """Faz previsões usando o modelo treinado."""
    if modelo is None or dados_entrada is None:
        return None
    try:
        previsoes = modelo.predict(dados_entrada)
        return previsoes
    except Exception as e:
        st.error(f"Erro ao fazer a previsão: {e}")
        return None

# Configurações do app
st.set_page_config(page_title="Previsão de Renda • EBAC", layout="wide")

# Estilo personalizado do segundo código
st.markdown("""
    <style>
    .main {
        background-color: #f0f8ff;  /* Fundo azul claro para um visual clean */
    }
    .stButton>button {
        background-color: #4CAF50;  /* Verde para botões, sinalizando ação positiva */
        color: white;
    }
    .stSuccess {
        background-color: #d4edda;  /* Fundo verde claro para mensagens de sucesso */
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar / Load
st.sidebar.title("Configurações")
st.sidebar.caption("O app tenta carregar previsao_de_renda.csv automaticamente. Se não achar, use o uploader.")

df = load_dataset()

st.title("📊 Previsão de Renda – Análise Completa e Modelagem")
st.markdown(
    "Este app reúne **todas** as análises e comparações do seu notebook, "
    "organizadas em abas, pronto para apresentação no Streamlit. Inclui previsão interativa de renda."
)

if df is None:
    st.info("Envie o arquivo CSV na barra lateral para começar.")
    st.stop()

# Preparos básicos
# Renda log
if "renda" in df.columns and pd.api.types.is_numeric_dtype(df["renda"]):
    df["renda_log"] = np.log(df["renda"].clip(lower=1e-9))

numeric_df = df.select_dtypes(include=[np.number]).copy()

# Pré-processamento e treinamento do modelo (do segundo código)
X, y, colunas_codificadas = preprocessar_dados(df)
modelo_treinado, X_teste, y_teste, nomes_features = treinar_modelo(X, y)

# Abas (adicionada uma aba para Previsão Interativa)
abas = st.tabs([
    "📥 Dados",
    "🔎 Exploração",
    "📈 Visualizações",
    "📊 Correlação",
    "🧪 Comparações solicitadas",
    "🤖 Modelagem (Regressão Linear)",
    "🔮 Previsão Interativa",
    "📝 Conclusões"
])

# ABA: Dados
with abas[0]:
    st.subheader("Amostra dos dados")
    st.dataframe(df.head(), use_container_width=True)

    c1, c2 = st.columns([1,1])
    with c1:
        st.subheader("Estatísticas descritivas")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    with c2:
        st.subheader("Info (schema)")
        st.code(read_info(df), language="text")

    st.markdown("---")
    st.subheader("Valores ausentes por coluna")
    na = df.isna().sum().sort_values(ascending=False)
    st.dataframe(na.to_frame("missing"), use_container_width=True)

# ABA: Exploração
with abas[1]:
    st.subheader("Exploração rápida com filtros")
    cols = st.multiselect("Selecione colunas para visualizar", df.columns.tolist(), default=df.columns[:10].tolist())
    st.dataframe(df[cols].head(30), use_container_width=True)

    st.markdown("---")
    st.subheader("Distribuição da variável *renda* e *renda_log*")
    if "renda" in df.columns and pd.api.types.is_numeric_dtype(df["renda"]):
        cc1, cc2 = st.columns(2)
        with cc1:
            fig, ax = plt.subplots()
            sns.histplot(df["renda"], kde=True, ax=ax)
            ax.set_title("Distribuição da renda")
            st.pyplot(fig)
        with cc2:
            fig, ax = plt.subplots()
            sns.boxplot(x=df["renda"], ax=ax)
            ax.set_title("Boxplot da renda (outliers)")
            st.pyplot(fig)

        if "renda_log" in df.columns:
            c3, c4 = st.columns(2)
            with c3:
                fig, ax = plt.subplots()
                sns.histplot(df["renda_log"], kde=True, ax=ax)
                ax.set_title("Distribuição da renda (log)")
                st.pyplot(fig)
            with c4:
                fig, ax = plt.subplots()
                sns.boxplot(x=df["renda_log"], ax=ax)
                ax.set_title("Boxplot da renda (log)")
                st.pyplot(fig)
    else:
        st.info("A coluna 'renda' não está disponível como numérica.")

# ABA: Visualizações
with abas[2]:
    st.subheader("Gráficos principais do projeto")

    # Scatter Idade vs Renda por Sexo
    if has_cols(df, ["idade", "renda", "sexo"]):
        st.caption("Idade vs Renda por Sexo")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="idade", y="renda", hue="sexo", alpha=0.6, ax=ax)
        ax.set_title("Relação entre Idade e Renda por Sexo")
        st.pyplot(fig)

    # Countplots categóricos
    cat_cols = [c for c in ["sexo", "educacao", "tipo_renda", "posse_de_veiculo", "posse_de_imovel"] if c in df.columns]
    if cat_cols:
        st.caption("Distribuições categóricas")
        grid = st.columns(2)
        for i, c in enumerate(cat_cols):
            with grid[i % 2]:
                fig, ax = plt.subplots()
                sns.countplot(data=df, x=c, ax=ax)
                ax.set_title(f"Distribuição de {c}")
                ax.tick_params(axis='x', rotation=35)
                st.pyplot(fig)

    # Treemap de tipo_renda (opcional)
    if "tipo_renda" in df.columns:
        squarify = try_import_squarify()
        if squarify is None:
            st.info("Para ver o Treemap, instale: `pip install squarify`")
        else:
            st.caption("Treemap – distribuição de Tipo de Renda")
            cont = df["tipo_renda"].value_counts().reset_index()
            cont.columns = ["tipo_renda", "contagem"]
            fig, ax = plt.subplots(figsize=(8,5))
            squarify.plot(sizes=cont["contagem"], label=cont["tipo_renda"], alpha=.8)
            ax.axis("off")
            ax.set_title("Treemap – Tipo de Renda")
            st.pyplot(fig)

    # Pairplot (pode ser pesado)
    st.markdown("""
- Outliers em renda
- Existem valores muito altos de renda, comuns em bases financeiras, que podem distorcer correlações e ajustes de modelos.
- O boxplot evidencia pontos extremos, sugerindo avaliação cuidadosa antes da modelagem.
""") 
    if numeric_df.shape[1] >= 2 and st.checkbox("Gerar Pairplot numérico (pode demorar)"):
        fig = sns.pairplot(numeric_df, diag_kind="kde", plot_kws={'alpha':0.5})
        st.pyplot(fig.fig)  # Alterado para fig.fig para corrigir renderização

# ABA: Correlação
with abas[3]:
    st.subheader("Matriz de correlação (variáveis numéricas)")
    if numeric_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm", ax=ax)
        ax.set_title("Heatmap de correlação")
        st.pyplot(fig)

        if "renda" in numeric_df.columns:
            st.subheader("Correlação de cada variável com 'renda'")
            corr = numeric_df.corr()["renda"].sort_values(ascending=False)
            st.dataframe(corr.to_frame("corr(renda)"), use_container_width=True)
    else:
        st.info("Não há colunas numéricas suficientes para calcular correlação.")

# ABA: Comparações solicitadas
with abas[4]:
    st.subheader("Comparações das variáveis mais correlacionadas com 'renda'")

    if "renda" in numeric_df.columns and numeric_df.shape[1] >= 3:
        corr = numeric_df.corr()["renda"].sort_values(ascending=False)
        top_vars = [v for v in corr.index if v != "renda"][:2]
        if len(top_vars) == 2:
            var1, var2 = top_vars
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots()
                sns.scatterplot(x=df[var1], y=df["renda"], alpha=0.6, ax=ax)
                ax.set_title(f"{var1} x renda")
                st.pyplot(fig)
            with c2:
                fig, ax = plt.subplots()
                sns.scatterplot(x=df[var2], y=df["renda"], alpha=0.6, ax=ax)
                ax.set_title(f"{var2} x renda")
                st.pyplot(fig)
        else:
            st.info("Não foi possível determinar duas variáveis além de 'renda' para a comparação.")
    else:
        st.info("São necessárias colunas numéricas (incluindo 'renda') para esta seção.")

    st.markdown("---")
    if "renda_log" in df.columns:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots()
            sns.boxplot(x=df["renda"], ax=ax)
            ax.set_title("Boxplot: renda")
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots()
            sns.boxplot(x=df["renda_log"], ax=ax)
            ax.set_title("Boxplot: renda (log)")
            st.pyplot(fig)

# ABA: Modelagem (Regressão)
with abas[5]:
    st.subheader("Pipeline de modelagem: Regressão Linear para prever 'renda'")

    if modelo_treinado is not None and X_teste is not None and y_teste is not None:
        y_pred = fazer_previsoes(modelo_treinado, X_teste)
        if y_pred is not None:
            mae = mean_absolute_error(y_teste, y_pred)
            mse = mean_squared_error(y_teste, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_teste, y_pred)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAE", f"{mae:,.2f}")
            m2.metric("MSE", f"{mse:,.2f}")
            m3.metric("RMSE", f"{rmse:,.2f}")
            m4.metric("R²", f"{r2:.3f}")

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Renda Real vs Renda Prevista")
                fig, ax = plt.subplots()
                sns.scatterplot(x=y_teste, y=y_pred, ax=ax, alpha=0.6)
                ax.set_xlabel("Renda real")
                ax.set_ylabel("Renda prevista")
                ax.set_title("Real vs Previsto")
                st.pyplot(fig)

            with c2:
                st.caption("Distribuição dos resíduos")
                residuos = y_teste - y_pred
                fig, ax = plt.subplots()
                sns.histplot(residuos, kde=True, ax=ax)
                ax.set_title("Resíduos do modelo")
                st.pyplot(fig)

            # Coeficientes (se aplicável)
            if hasattr(modelo_treinado, "coef_"):
                coef_df = pd.DataFrame({"Variável": nomes_features, "Coeficiente": modelo_treinado.coef_}).sort_values(
                    by="Coeficiente", key=lambda s: s.abs(), ascending=False
                )
                st.subheader("Coeficientes (ordenados por magnitude)")
                st.dataframe(coef_df, use_container_width=True)
        else:
            st.warning("Não foi possível gerar previsões para avaliação.")
    else:
        st.warning("Modelo não treinado. Verifique os dados.")

# ABA: Previsão Interativa (do segundo código)
with abas[6]:
    st.subheader("🤑 Previsão de Renda Interativa")
    st.markdown("Insira os dados do cliente abaixo e clique em 'Prever Renda'.")

    if df is not None:
        # Opções únicas para selects
        opcoes_sexo = df['sexo'].unique().tolist()
        opcoes_tipo_renda = df['tipo_renda'].unique().tolist()
        opcoes_educacao = df['educacao'].unique().tolist()
        opcoes_estado_civil = df['estado_civil'].unique().tolist()
        opcoes_tipo_residencia = df['tipo_residencia'].unique().tolist()

        # Inputs
        idade = st.slider("Idade", min_value=int(df['idade'].min()), max_value=int(df['idade'].max()), value=int(df['idade'].mean()), help="Idade do cliente em anos.")
        tempo_emprego = st.slider("Tempo de Emprego (anos)", min_value=0.0, max_value=float(df['tempo_emprego'].max()), value=float(df['tempo_emprego'].mean()), help="Tempo no emprego atual.")
        qtd_filhos = st.slider("Quantidade de Filhos", min_value=0, max_value=int(df['qtd_filhos'].max()), value=int(df['qtd_filhos'].mean()), help="Número de filhos.")
        qt_pessoas_residencia = st.slider("Pessoas na Residência", min_value=1.0, max_value=float(df['qt_pessoas_residencia'].max()), value=float(df['qt_pessoas_residencia'].mean()), help="Total de pessoas na casa.")

        sexo = st.selectbox("Sexo", opcoes_sexo, help="Sexo do cliente (F ou M).")
        posse_veiculo = st.selectbox("Posse de Veículo", [True, False], help="Possui veículo?")
        posse_imovel = st.selectbox("Posse de Imóvel", [True, False], help="Possui imóvel?")
        tipo_renda = st.selectbox("Tipo de Renda", opcoes_tipo_renda, help="Fonte principal de renda.")
        educacao = st.selectbox("Nível de Educação", opcoes_educacao, help="Grau de escolaridade.")
        estado_civil = st.selectbox("Estado Civil", opcoes_estado_civil, help="Situação marital.")
        tipo_residencia = st.selectbox("Tipo de Residência", opcoes_tipo_residencia, help="Tipo de moradia.")

        # Prepara dados de entrada
        if nomes_features is not None:
            dados_entrada_dict = {col: 0 for col in nomes_features}
            dados_entrada_dict['idade'] = idade
            dados_entrada_dict['tempo_emprego'] = tempo_emprego
            dados_entrada_dict['qtd_filhos'] = qtd_filhos
            dados_entrada_dict['qt_pessoas_residencia'] = qt_pessoas_residencia

            # Manipula features booleanas
            if 'sexo_M' in dados_entrada_dict:
                dados_entrada_dict['sexo_M'] = 1 if sexo == 'M' else 0
            if 'posse_de_veiculo_True' in dados_entrada_dict:
                dados_entrada_dict['posse_de_veiculo_True'] = 1 if posse_veiculo else 0
            if 'posse_de_imovel_True' in dados_entrada_dict:
                dados_entrada_dict['posse_de_imovel_True'] = 1 if posse_imovel else 0
            if 'mau_True' in dados_entrada_dict:
                dados_entrada_dict['mau_True'] = 0  # Assumindo valor padrão

            # Manipula features categóricas codificadas
            for cat, valor in {'tipo_renda': tipo_renda, 'educacao': educacao, 'estado_civil': estado_civil, 'tipo_residencia': tipo_residencia}.items():
                chave = f'{cat}_{valor}'
                if chave in dados_entrada_dict:
                    dados_entrada_dict[chave] = 1

            df_entrada = pd.DataFrame([dados_entrada_dict])[nomes_features]

            # Botão de previsão
            if st.button("Prever Renda"):
                previsao_renda = fazer_previsoes(modelo_treinado, df_entrada)
                if previsao_renda is not None:
                    st.success(f"**Previsão de Renda:** R$ {previsao_renda[0]:,.2f}")
                else:
                    st.warning("Não foi possível fazer a previsão. Verifique os dados.")

# ABA: Conclusões
with abas[7]:
    st.subheader("Insights e Conclusões do Projeto")
    st.markdown("""
**Outliers em `renda`**  
– Existem valores muito altos na renda, comuns em bases financeiras; eles podem distorcer correlações/modelos.  
– O **boxplot** evidencia esses pontos.

**Log-transform de `renda`**  
– A transformação `log(renda)` reduz assimetria e aproxima de uma distribuição mais estável.  
– Melhora a linearidade e reduz impacto de outliers, beneficiando a **regressão linear**.

**Correlação**  
– A matriz de correlação aponta as variáveis com maior associação com `renda`.  
– As duas mais correlacionadas foram usadas para **gráficos de dispersão** dedicados.

**Recomendações**  
– Avaliar *feature engineering* (ex.: interação, binning, Winsorization).  
– Considerar modelos robustos a outliers (ex.: Huber) ...
""")