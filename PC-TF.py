import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from dowhy.gcm.falsify import falsify_graph
from dowhy.gcm.util.general import set_random_seed
from matplotlib.backends.backend_pdf import PdfPages
from networkx.drawing.nx_pydot import read_dot

set_random_seed(1332)

GRAPH_FILE = "PC_output_graph.gml"
DATA_FILE = "dataset_270411_final_ingles_sin_GPSR.csv"
OUTPUT_DIR = "resultados_falsificacion"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def pearson_independence_test(X, Y, condition_set=None):
    if condition_set is not None and len(condition_set) > 0:
        return 1.0
    _, p = pearsonr(X, Y)
    return p

print("Cargando grafo...")
g = nx.read_gml(GRAPH_FILE)
g = nx.DiGraph(g)

print(f"Nodos: {g.number_of_nodes()}")
print(f"Aristas: {g.number_of_edges()}")

print("Verificando si el grafo es DAG...")
g_dag = g.copy()

while not nx.is_directed_acyclic_graph(g_dag):
    cycle = list(nx.simple_cycles(g_dag))[0]
    u, v = cycle[-1], cycle[0]
    print(f"Eliminando arista {u} -> {v}")
    g_dag.remove_edge(u, v)

print("DAG válido obtenido")

nx.write_gml(g_dag, os.path.join(OUTPUT_DIR, "PC_DAG_final.gml"))

print("Cargando dataset...")
data = pd.read_csv(DATA_FILE, sep=';')

data = data[list(g_dag.nodes())]
data = data.dropna()

MAX_SAMPLES = 1000

if len(data) > MAX_SAMPLES:
    data = data.sample(n=MAX_SAMPLES, random_state=1332)

data = data.reset_index(drop=True)

print(f"Filas usadas para falsificación: {len(data)}")

print("Ejecutando test de falsificación...")

result = falsify_graph(
    causal_graph=g_dag,
    data=data,
    independence_test=pearson_independence_test,
    significance_level=0.05,
    n_permutations=1000,
    plot_histogram=True,
    n_jobs=1,
    plot_kwargs={'savepath': 'falsification_plot_pc.png'}  
)

plt.close()

print("Test de falsificación ejecutado correctamente")

print("\n===== RESUMEN =====")
print(result)
