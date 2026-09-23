import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import networkx as nx
import pydot

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils

np.set_printoptions(precision=3, suppress=True)
np.random.seed(0)
TARGET = "SCORE_NEM"
print("Cargando dataset")
data_mpg = pd.read_csv(
    "dataset_270411_final_ingles_sin_GPSR.csv",
    sep=";",
    low_memory=False
)

data_mpg.columns = data_mpg.columns.str.strip().str.replace('"', '', regex=False)

if TARGET not in data_mpg.columns:
    raise Exception(f"TARGET '{TARGET}' no existe")
print("Procesando datos...")
data_mpg = data_mpg.apply(pd.to_numeric, errors='coerce')

# primero convertir, luego aplico dropna no olvidar
data_mpg = data_mpg.dropna()

# REORDENAR PARA EL TARGET
cols = [c for c in data_mpg.columns if c != TARGET] + [TARGET]
data_mpg = data_mpg[cols]
labels = list(data_mpg.columns)
data = data_mpg.to_numpy(dtype=np.float32)

# Configuración del PC
print("\nEjecutando PC (optimizado)...")

cg = pc(
    data,
    alpha=0.05,             # restrictivo estándar
    indep_test='fisherz',
    stable=True,
    uc_rule=0,              # más rápido que 1
    uc_priority=0,
    verbose=False,
    show_progress=False
)

print("PC finalizado.")


# GRAFO
print("Construyendo grafo...")
pyd = GraphUtils.to_pydot(cg.G, labels=labels)
pyd.set_graph_defaults(
    rankdir="TB",
    splines="true",
    overlap="false"
)

pyd.set_node_defaults(
    shape="ellipse",
    fontsize="10"
)

pyd.set_edge_defaults(
    fontsize="8",
    penwidth="1"
)


# EXPORTAR: salidar en dot, png y pdf
dot_filename = "PC_output_graph.dot"
png_filename = "PC_output_graph.png"
pdf_filename = "PC_output_graph.pdf"

print("\nExportando...")

pyd.write_raw(dot_filename)
pyd.write_png(png_filename)
pyd.write_pdf(pdf_filename)

print("Listo:")
print(png_filename)


# MOSTRAR png
img = mpimg.imread(png_filename)
plt.figure(figsize=(14, 14))
plt.imshow(img)
plt.axis('off')
plt.tight_layout()
plt.show()

# Aqui aplicar NETWORKX
print("\nAnalizando TARGET...")

G = nx.DiGraph()
G.add_nodes_from(labels)

graph = cg.G.graph
n = len(labels)

for i in range(n):
    for j in range(n):

        if graph[i, j] == -1 and graph[j, i] == 1:
            G.add_edge(labels[i], labels[j])

if TARGET in G:

    print("\nPADRES:", list(G.predecessors(TARGET)))
    print("HIJOS:", list(G.successors(TARGET)))
    print("ANCESTROS:", list(nx.ancestors(G, TARGET)))
    print("DESCENDIENTES:", list(nx.descendants(G, TARGET)))

else:
    print("TARGET no encontrado.")
