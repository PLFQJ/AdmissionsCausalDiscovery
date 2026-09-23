import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import networkx as nx
import re
import os

# CONFIGURACIÓN
DOT_FILE = "PC_output_graph.dot"
CSV_FILE = "dataset_270411_final_ingles.csv"

TARGET_LABEL = "SCORE_NEM"
TREATMENT_LABEL = "WORK"

TREATMENT_0 = 0
TREATMENT_1 = 1

# 1. LECTOR DOT (CORRECTO: ID + LABEL)
def load_dot_graph(path):

    G = nx.DiGraph()
    id_to_label = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:

            line = line.strip()

            
            # NODOS
            
            node_match = re.findall(r"(\d+)\s*\[label=(.*?)\];", line)

            for node_id, label in node_match:
                label = label.replace('"', '').strip()
                id_to_label[node_id] = label
                G.add_node(node_id, label=label)

            
            # EDGES
            
            edge_match = re.findall(r"(\d+)\s*->\s*(\d+)", line)

            for u, v in edge_match:
                G.add_edge(u, v)

    return G, id_to_label


G, id_to_label = load_dot_graph(DOT_FILE)

label_to_id = {v: k for k, v in id_to_label.items()}


# 2. MAPEO VARIABLES
if TARGET_LABEL not in label_to_id:
    raise Exception("TARGET no existe en DAG")

if TREATMENT_LABEL not in label_to_id:
    raise Exception("TREATMENT no existe en DAG")

target_node = label_to_id[TARGET_LABEL]
treat_node = label_to_id[TREATMENT_LABEL]

print("\nTarget node:", target_node)
print("Treatment node:", treat_node)


# 3. BACKDOOR SET (PADRES DEL TARGET)
parents = list(G.predecessors(target_node))
parents_labels = [id_to_label[p] for p in parents if p in id_to_label]

print("\nBackdoor candidates:", parents_labels)

# 4. CARGA DATASET
df = pd.read_csv(CSV_FILE, sep=";")
df.columns = [c.strip().replace('"', '') for c in df.columns]

# variables válidas
controls = [c for c in parents_labels if c in df.columns]

if len(controls) == 0:
    raise Exception("No hay variables de ajuste válidas")

df_model = df[[TARGET_LABEL] + controls].dropna()

# 5. MODELO CAUSAL (BACKDOOR OLS)
X = df_model[controls]
y = df_model[TARGET_LABEL]

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print("\n==============================")
print("MODELO CAUSAL AJUSTADO")
print("==============================")
print(model.summary())

# 6. OBSERVADO
df_model["Observed"] = model.predict(X)

# 7. CONTRAFACTUAL (INTERVENCIÓN DO(WORK))
X_cf = X.copy()

if TREATMENT_LABEL in X_cf.columns:
    X_cf[TREATMENT_LABEL] = TREATMENT_1

df_model["Counterfactual"] = model.predict(X_cf)

df_model["Effect"] = df_model["Counterfactual"] - df_model["Observed"]

# 8. MÉTRICAS
ate = df_model["Effect"].mean()

print("\n==============================")
print("RESULTADOS CAUSALES")
print("==============================")
print("ATE:", ate)

# 9. EXPORT CSV
df_model.to_csv(
    "causal_backdoor_results.csv",
    sep=";",
    index=False
)

# 10. GRÁFICO 1 - MEDIA (PNG + PDF)
plt.figure(figsize=(8, 6))

plt.bar(
    ["Observed", "Counterfactual (do(WORK=1))"],
    [df_model["Observed"].mean(), df_model["Counterfactual"].mean()],
    edgecolor="black"
)

plt.title("Causal Effect of WORK on SCORE_NEM", fontsize=14)
plt.ylabel("SCORE_NEM (Predicted)", fontsize=12)
plt.xlabel("Scenario", fontsize=12)

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig("fig1_causal_mean.png", dpi=600)
plt.savefig("fig1_causal_mean.pdf")

plt.close()

# 11. GRÁFICO 2 - DISTRIBUCIÓN
plt.figure(figsize=(8, 6))

plt.hist(df_model["Observed"], bins=30, alpha=0.6, label="Observed")
plt.hist(df_model["Counterfactual"], bins=30, alpha=0.6, label="Counterfactual")

plt.title("Distribution: Observed vs Counterfactual", fontsize=14)
plt.xlabel("SCORE_NEM", fontsize=12)
plt.ylabel("Frequency", fontsize=12)

plt.legend()

plt.tight_layout()

plt.savefig("fig2_distribution.png", dpi=600)
plt.savefig("fig2_distribution.pdf")

plt.close()

# 12. GRÁFICO 3 - EFECTOS INDIVIDUALES
plt.figure(figsize=(8, 6))

plt.hist(df_model["Effect"], bins=30, color="gray", edgecolor="black")

plt.title("Individual Treatment Effects (ITE)", fontsize=14)
plt.xlabel("Effect of WORK on SCORE_NEM", fontsize=12)
plt.ylabel("Frequency", fontsize=12)

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig("fig3_ite.png", dpi=600)
plt.savefig("fig3_ite.pdf")

plt.close()

# 13. FINAL
print("\n==============================")
print("PROCESO FINALIZADO")
print("==============================")

print("\nArchivos generados:")
print("- causal_backdoor_results.csv")
print("- fig1_causal_mean.png / .pdf")
print("- fig2_distribution.png / .pdf")
print("- fig3_ite.png / .pdf")
