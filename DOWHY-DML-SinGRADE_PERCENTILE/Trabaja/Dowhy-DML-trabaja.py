import numpy as np
import pandas as pd
import dowhy
from dowhy import CausalModel
import econml
import warnings
warnings.filterwarnings('ignore')
import networkx as nx
import re
import os

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from econml.inference import BootstrapInference


# CONFIGURACION
CSV_FILE = "dataset_270411_final_ingles_sin_GPSR.csv"
DOT_FILE = "PC_output_graph.dot"
TREATMENT = "WORK"
TARGET = "SCORE_NEM"

# El DOT PC entregado contiene relaciones que, al interpretarlas
# literalmente como flechas, forman ciclos. DoWhy necesita un DAG.
# "auto" conserva las aristas en el orden del DOT y rechaza solo
# las que introducirían un ciclo. Las aristas rechazadas se informan.
PC_CYCLE_POLICY = "auto"


# CABECERA
print("\n" + "=" * 70)
print("ANALISIS CAUSAL CON DoWhy")
print("=" * 70)

print("\nVersiones instaladas:")
print("DoWhy :", dowhy.__version__)
print("EconML:", econml.__version__)


# FUNCIONES AUXILIARES
def limpiar_columnas(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace('"', '', regex=False)
    )
    return df


def cargar_pc_dot(dot_file, columnas_csv):
    """
    Lee directamente el DOT de PC sin nx.nx_pydot.read_dot().

    El PC.DOT suministrado tiene nodos numericos con labels:
        0 [label="GRADUATION_YEAR"];
        1 [label="GRADE_PERCENTILE"];

    y relaciones:
        1 -> 0 [dir=both, arrowtail=none, arrowhead=normal];

    Se traducen los IDs a nombres del CSV.
    """

    print("\n" + "=" * 70)
    print("CARGANDO DAG PC")
    print("=" * 70)

    print("\nArchivo DOT:", dot_file)

    if not os.path.isfile(dot_file):
        raise FileNotFoundError(
            f"\nNo existe el archivo DOT:\n{os.path.abspath(dot_file)}"
        )

    with open(dot_file, "r", encoding="utf-8") as f:
        dot_text = f.read()

    print("\nArchivo DOT leído correctamente.")

    # --------------------------------------------------------
    # ID -> nombre real de variable
    # --------------------------------------------------------
    node_map = {}

    node_pattern = re.compile(
        r'(?m)^\s*(\d+)\s+\[label=(?:"([^"]+)"|([^;\]]+))\];'
    )

    for m in node_pattern.finditer(dot_text):
        node_id = m.group(1)
        node_name = (
            m.group(2)
            if m.group(2) is not None
            else m.group(3)
        )
        node_name = node_name.strip().strip('"')
        node_map[node_id] = node_name

    if not node_map:
        raise ValueError(
            "\nNo se pudieron identificar los nodos del pc.dot."
        )

    print("\nVariables encontradas en PC:")
    for node_id, node_name in node_map.items():
        print(f"  {node_id:>3} -> {node_name}")

    # Extraer relaciones
    edge_pattern = re.compile(
        r'(?m)^\s*(\d+)\s*->\s*(\d+)\s*\[([^\]]+)\];'
    )

    directed_edges = []
    undirected_edges = []

    for m in edge_pattern.finditer(dot_text):

        source_id = m.group(1)
        target_id = m.group(2)
        attrs = m.group(3)

        if (
            source_id not in node_map
            or target_id not in node_map
        ):
            continue

        source = node_map[source_id]
        target = node_map[target_id]

        if (
            source not in columnas_csv
            or target not in columnas_csv
        ):
            continue

        if "arrowhead=normal" in attrs:
            directed_edges.append((source, target))

        elif "arrowhead=none" in attrs:
            undirected_edges.append((source, target))

    print("\nRelaciones encontradas en PC:")

    for source, target in directed_edges:
        print(f"  {source} -> {target}")

    for source, target in undirected_edges:
        print(f"  {source} --- {target}")

    print("\nRelaciones dirigidas:", len(directed_edges))
    print("Relaciones no orientadas:", len(undirected_edges))

    if not directed_edges:
        raise ValueError(
            "\nNo se encontraron relaciones dirigidas en pc.dot."
        )

    # --------------------------------------------------------
    # Construir DAG compatible con DoWhy
    #
    # El DOT suministrado contiene ciclos si se toman todas
    # sus flechas literalmente. DoWhy requiere un DAG.
    #
    # Conservamos las direcciones escritas en el DOT y
    # se descarta solamente una arista cuando su inclusión
    # introduciría un ciclo.
    #
    # IMPORTANTE:
    # Esto produce un DAG operacional para DoWhy, pero las
    # aristas descartadas deben revisarse metodológicamente.
    # --------------------------------------------------------
    graph = nx.DiGraph()

    for node_name in node_map.values():
        if node_name in columnas_csv:
            graph.add_node(node_name)

    rejected_cycles = []

    for source, target in directed_edges:

        if source == target:
            rejected_cycles.append(
                (source, target, "auto-loop")
            )
            continue

        if graph.has_edge(source, target):
            continue

        # Si existe target -> ... -> source,
        # source -> target cerraría un ciclo.
        if nx.has_path(graph, target, source):

            rejected_cycles.append(
                (source, target, "introduce-ciclo")
            )

            continue

        graph.add_edge(source, target)


    # Resumen
    print("\n" + "-" * 70)
    print("GRAFO COMPATIBLE CON DoWhy")
    print("-" * 70)

    print("Nodos:", graph.number_of_nodes())
    print("Aristas conservadas:", graph.number_of_edges())
    print("Aristas dirigidas originales:", len(directed_edges))
    print("Aristas no orientadas:", len(undirected_edges))
    print(
        "Aristas descartadas por ciclos:",
        len(rejected_cycles)
    )

    if rejected_cycles:

        print(
            "\nAristas descartadas porque "
            "introducirían ciclos:"
        )

        for source, target, reason in rejected_cycles:
            print(
                f"  {source} -> {target} [{reason}]"
            )

    if not nx.is_directed_acyclic_graph(graph):

        raise RuntimeError(
            "\nNo fue posible construir un DAG "
            "compatible con DoWhy."
        )

    print("\nDAG verificado correctamente.")

    if TREATMENT not in graph.nodes:

        raise ValueError(
            f"\nLa variable tratamiento "
            f"'{TREATMENT}' no aparece en PC."
        )

    if TARGET not in graph.nodes:

        raise ValueError(
            f"\nLa variable resultado "
            f"'{TARGET}' no aparece en PC."
        )

    return (
        graph,
        directed_edges,
        undirected_edges,
        rejected_cycles
    )


# CARGA DEL DATASET
print("\n" + "=" * 70)
print("CARGA DEL DATASET")
print("=" * 70)

print("\nCargando dataset...")

if not os.path.isfile(CSV_FILE):

    raise FileNotFoundError(
        f"\nNo existe el CSV:\n"
        f"{os.path.abspath(CSV_FILE)}"
    )

data = pd.read_csv(
    CSV_FILE,
    sep=';'
)

data = limpiar_columnas(data)

print("\nColumnas del dataset:")
print(list(data.columns))

if TARGET not in data.columns:

    raise Exception(
        f"\nTARGET '{TARGET}' no existe en el CSV."
    )

if TREATMENT not in data.columns:

    raise Exception(
        f"\nTREATMENT '{TREATMENT}' no existe en el CSV."
    )

print("\nProcesando...")

data = data.apply(
    pd.to_numeric,
    errors='coerce'
)

data = data.dropna().reset_index(
    drop=True
)

print(
    "\nNumero de observaciones:",
    len(data)
)

if len(data) == 0:

    raise Exception(
        "El dataset quedó sin observaciones "
        "después de dropna()."
    )


# CARGAR PC DOT
(
    graph,
    pc_directed_edges,
    pc_undirected_edges,
    pc_rejected_edges
) = cargar_pc_dot(
    DOT_FILE,
    set(data.columns)
)


# INFORMACION DEL GRAFO
print("\n" + "=" * 70)
print("ESTRUCTURA CAUSAL UTILIZADA POR DoWhy")
print("=" * 70)

print("\nTratamiento:", TREATMENT)
print("Resultado  :", TARGET)

print("\nAristas utilizadas:")

for source, target in graph.edges():

    print(
        f"  {source} -> {target}"
    )


# CREAR MODELO DoWhy
print("\n" + "=" * 70)
print("CREANDO MODELO DoWhy")
print("=" * 70)

model = CausalModel(
    data=data,
    treatment=TREATMENT,
    outcome=TARGET,
    graph=graph
)

print(
    "\nModelo DoWhy creado correctamente."
)


# IDENTIFICAR EFECTO
identified_estimand = model.identify_effect(
    proceed_when_unidentifiable=True
)

print("\n" + "=" * 70)
print("IDENTIFICACIÓN DEL EFECTO CAUSAL")
print("=" * 70)

print(
    identified_estimand
)


# LINEAR MODEL
print("\n\n\n\n---- Linear Model ----")

linear_estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression",
    control_value=0,
    treatment_value=1
)

print(
    linear_estimate
)

print(
    "INTERPRETACIÓN DEL RESULTADO"
)

print(
    "METODO ES: "
    "method_name=backdoor.linear_regression"
)

print(
    "\n" + "=" * 70
)

print(
    "INTERPRETACIÓN"
)

print(
    "=" * 70
)

try:

    ate = linear_estimate.value

    print(
        f"\nATE estimado = {ate:.6f}"
    )

    print(
        "\n¿Qué representa este valor?"
    )

    print(
        "----------------------------------------"
    )

    print(
        "El ATE (Average Treatment Effect) "
        "representa el cambio"
    )

    print(
        "promedio esperado en la variable "
        "resultado cuando"
    )

    print(
        "el tratamiento cambia desde el "
        "valor de control"
    )

    print(
        "(0) al valor de tratamiento (1), "
        "manteniendo"
    )

    print(
        "ajustadas las variables de "
        "confusión del DAG."
    )

    if ate > 0:

        print(
            "\nInterpretación:"
        )

        print(
            "El tratamiento aumenta el resultado "
            "en promedio "
            f"{abs(ate):.4f} unidades."
        )

    elif ate < 0:

        print(
            "\nInterpretación:"
        )

        print(
            "El tratamiento disminuye el resultado "
            "en promedio "
            f"{abs(ate):.4f} unidades."
        )

    else:

        print(
            "\nInterpretación:"
        )

        print(
            "No se observa un efecto promedio "
            "del tratamiento."
        )

    magnitud = abs(ate)

    if magnitud < 0.10:
        nivel = "muy pequeño"

    elif magnitud < 0.50:
        nivel = "pequeño"

    elif magnitud < 1.00:
        nivel = "moderado"

    elif magnitud < 2.00:
        nivel = "grande"

    else:
        nivel = "muy grande"

    print(
        f"\nMagnitud del efecto: {nivel}"
    )

    print(
        "\nConclusión:"
    )

    print(
        "Este valor corresponde al efecto "
        "causal promedio (ATE)"
    )

    print(
        "estimado mediante Regresión Lineal "
        "usando el criterio"
    )

    print(
        "Backdoor para controlar los factores "
        "de confusión."
    )

    print(
        "La interpretación asume que el DAG "
        "especificado es correcto"
    )

    print(
        "y que no existen confusores no "
        "observados relevantes."
    )

except Exception as e:

    print(
        "No fue posible interpretar "
        "automáticamente el resultado."
    )

    print(e)

print(
    "FIN INTERPRETACION METODO ES: "
    "method_name=backdoor.linear_regression\n\n\n\n"
)



# DOUBLE MACHINE LEARNING
print(
    "---- EconML methods ----"
)

print(
    "---- EconML methods ----efecto promedio"
)

dml_estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.econml.dml.DML",
    control_value=0,
    treatment_value=1,
    confidence_intervals=False,
    method_params={
        "init_params": {
            "model_y":
                GradientBoostingRegressor(),

            "model_t":
                GradientBoostingRegressor(),

            "model_final":
                LassoCV(
                    fit_intercept=False
                ),

            "featurizer":
                PolynomialFeatures(
                    degree=1,
                    include_bias=False
                )
        },

        "fit_params": {}
    }
)

print(
    dml_estimate
)


print(
    "\n" + "=" * 70
)

print(
    "INTERPRETACIÓN DEL DOUBLE MACHINE LEARNING"
)

print(
    "=" * 70
)

try:

    ate = dml_estimate.value

    print(
        f"\nATE estimado (DML): {ate:.6f}"
    )

    print(
        "\n¿Qué significa este resultado?"
    )

    print(
        "---------------------------------------"
    )

    print(
        "Double Machine Learning estima el "
        "efecto causal"
    )

    print(
        "eliminando primero el efecto de las "
        "variables"
    )

    print(
        "de confusión mediante modelos de "
        "Machine Learning."
    )

    print(
        "Posteriormente estima el efecto causal "
        "sobre"
    )

    print(
        "la información residual "
        "(ortogonalización)."
    )

    print(
        "\nEste procedimiento reduce el sesgo cuando"
    )

    print(
        "las relaciones entre variables son "
        "complejas"
    )

    print(
        "o altamente no lineales."
    )

    if ate > 0:

        print(
            "\nInterpretación:"
        )

        print(
            "El tratamiento incrementa el resultado "
            f"en promedio {abs(ate):.4f} unidades."
        )

    elif ate < 0:

        print(
            "\nInterpretación:"
        )

        print(
            "El tratamiento reduce el resultado "
            f"en promedio {abs(ate):.4f} unidades."
        )

    else:

        print(
            "\nInterpretación:"
        )

        print(
            "No se detecta un efecto causal promedio."
        )

    efecto = abs(ate)

    if efecto < 0.10:
        nivel = "muy pequeño"

    elif efecto < 0.50:
        nivel = "pequeño"

    elif efecto < 1:
        nivel = "moderado"

    elif efecto < 2:
        nivel = "grande"

    else:
        nivel = "muy grande"

    print(
        f"\nMagnitud del efecto: {nivel}"
    )

    print(
        "\n¿Por qué utilizar DML?"
    )

    print(
        "-----------------------"
    )

    print(
        "✓ Utiliza modelos de Machine Learning para"
    )

    print(
        "  modelar el tratamiento y el resultado."
    )

    print(
        "✓ Reduce el sesgo por errores de especificación."
    )

    print(
        "✓ Funciona adecuadamente cuando existen"
    )

    print(
        "  relaciones no lineales."
    )

    print(
        "✓ Es apropiado para conjuntos de datos"
    )

    print(
        "  con muchas variables."
    )

    print(
        "\nConclusión:"
    )

    print(
        "El valor obtenido corresponde al Average"
    )

    print(
        "Treatment Effect (ATE) estimado mediante"
    )

    print(
        "Double Machine Learning."
    )

    print(
        "Si el DAG es correcto y se cumplen los"
    )

    print(
        "supuestos de identificabilidad, este valor"
    )

    print(
        "representa el efecto causal promedio del"
    )

    print(
        "tratamiento sobre la variable resultado."
    )

except Exception as e:

    print(
        "No fue posible interpretar automáticamente "
        "el resultado."
    )

    print(e)

print(
    "FIN INTERPRETACIÓN DEL DOUBLE MACHINE LEARNING\n\n\n\n"
)



# CATE: SCORE_NEM > 1
print(
    "---- EconML methods ----efecto TARGET"
)

dml_estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.econml.dml.DML",
    control_value=0,
    treatment_value=1,
    target_units=lambda df: df["SCORE_NEM"] > 1,
    confidence_intervals=False,
    method_params={
        "init_params": {
            "model_y":
                GradientBoostingRegressor(),

            "model_t":
                GradientBoostingRegressor(),

            "model_final":
                LassoCV(
                    fit_intercept=False
                ),

            "featurizer":
                PolynomialFeatures(
                    degree=1,
                    include_bias=False
                )
        },

        "fit_params": {}
    }
)

print(
    dml_estimate
)


print(
    "\n" + "=" * 70
)

print(
    "INTERPRETACIÓN DEL CATE"
)

print(
    "=" * 70
)

try:

    cate = dml_estimate.value

    print(
        f"\nCATE estimado = {cate:.6f}"
    )

    print(
        "\n¿Qué significa este resultado?"
    )

    print(
        "------------------------------------------"
    )

    print(
        "A diferencia del ATE, el CATE estima el"
    )

    print(
        "efecto causal únicamente para un subconjunto"
    )

    print(
        "de la población."
    )

    print(
        "\nEn este análisis el subconjunto corresponde a:"
    )

    print(
        "    SCORE_NEM > 1"
    )

    print(
        "\nEs decir, el efecto causal se calcula"
    )

    print(
        "solamente para las observaciones que"
    )

    print(
        "cumplen dicha condición."
    )

    if cate > 0:

        print(
            "\nInterpretación:"
        )

        print(
            "Para este grupo específico, el tratamiento "
            f"aumenta el resultado en promedio {abs(cate):.4f} unidades."
        )

    elif cate < 0:

        print(
            "\nInterpretación:"
        )

        print(
            "Para este grupo específico, el tratamiento "
            f"disminuye el resultado en promedio {abs(cate):.4f} unidades."
        )

    else:

        print(
            "\nInterpretación:"
        )

        print(
            "No se observa un efecto causal promedio"
        )

        print(
            "para este subconjunto."
        )

    efecto = abs(cate)

    if efecto < 0.10:
        nivel = "muy pequeño"

    elif efecto < 0.50:
        nivel = "pequeño"

    elif efecto < 1:
        nivel = "moderado"

    elif efecto < 2:
        nivel = "grande"

    else:
        nivel = "muy grande"

    print(
        f"\nMagnitud del efecto: {nivel}"
    )

    print(
        "\n¿Por qué calcular un CATE?"
    )

    print(
        "--------------------------"
    )

    print(
        "No todos los individuos responden"
    )

    print(
        "de la misma forma al tratamiento."
    )

    print(
        "El CATE permite identificar"
    )

    print(
        "heterogeneidad en los efectos causales."
    )

    print(
        "Es posible que un tratamiento tenga"
    )

    print(
        "un efecto positivo para algunos grupos"
    )

    print(
        "y negativo para otros."
    )

    print(
        "\nConclusión:"
    )

    print(
        "El valor obtenido corresponde al efecto"
    )

    print(
        "causal promedio únicamente para las"
    )

    print(
        "observaciones que cumplen la condición"
    )

    print(
        "SCORE_NEM > 1."
    )

    print(
        "No debe interpretarse como el efecto"
    )

    print(
        "causal promedio de toda la población."
    )

except Exception as e:

    print(
        "No fue posible interpretar automáticamente "
        "el resultado."
    )

    print(e)

print(
    "FIN INTERPRETACIÓN DEL CATE DOUBLE MACHINE LEARNING\n\n\n\n"
)


# CATE INDIVIDUAL
print(
    "---- EconML methods ----efecto TARGET en 1"
)

print(
    "True causal estimate is",
    data["SCORE_NEM"]
)

dml_estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.econml.dml.DML",
    control_value=0,
    treatment_value=1,
    target_units=1,
    confidence_intervals=False,
    method_params={
        "init_params": {
            "model_y":
                GradientBoostingRegressor(),

            "model_t":
                GradientBoostingRegressor(),

            "model_final":
                LassoCV(
                    fit_intercept=False
                ),

            "featurizer":
                PolynomialFeatures(
                    degree=1,
                    include_bias=True
                )
        },

        "fit_params": {}
    }
)

print(
    dml_estimate
)


print(
    "\n" + "=" * 70
)

print(
    "INTERPRETACIÓN DEL CATE INDIVIDUAL"
)

print(
    "=" * 70
)

try:

    cate = dml_estimate.value

    print(
        f"\nEfecto causal estimado = {cate:.6f}"
    )

    print(
        "\n¿Qué significa este resultado?"
    )

    print(
        "---------------------------------------------"
    )

    print(
        "En este análisis se utilizó:"
    )

    print(
        "target_units = 1"
    )

    print(
        "Por lo tanto, el efecto estimado no representa"
    )

    print(
        "el promedio para toda la población (ATE), sino"
    )

    print(
        "el efecto causal para una unidad o instancia"
    )

    print(
        "específica, según la definición utilizada por"
    )

    print(
        "DoWhy/EconML."
    )

    if cate > 0:

        print(
            "\nInterpretación:"
        )

        print(
            "Para esta unidad, recibir el tratamiento "
            "aumenta el resultado en aproximadamente "
            f"{abs(cate):.4f} unidades."
        )

    elif cate < 0:

        print(
            "\nInterpretación:"
        )

        print(
            "Para esta unidad, recibir el tratamiento "
            "disminuye el resultado en aproximadamente "
            f"{abs(cate):.4f} unidades."
        )

    else:

        print(
            "\nInterpretación:"
        )

        print(
            "No se observa un efecto causal para"
        )

        print(
            "la unidad evaluada."
        )

    efecto = abs(cate)

    if efecto < 0.10:
        nivel = "muy pequeño"

    elif efecto < 0.50:
        nivel = "pequeño"

    elif efecto < 1:
        nivel = "moderado"

    elif efecto < 2:
        nivel = "grande"

    else:
        nivel = "muy grande"

    print(
        f"\nMagnitud del efecto: {nivel}"
    )

    print(
        "\nCaracterísticas del método:"
    )

    print(
        "---------------------------"
    )

    print(
        "• Utiliza Double Machine Learning."
    )

    print(
        "• Ajusta automáticamente el efecto"
    )

    print(
        "  de las covariables mediante"
    )

    print(
        "  algoritmos de Machine Learning."
    )

    print(
        "• Permite estimar efectos causales"
    )

    print(
        "  heterogéneos entre individuos."
    )

    print(
        "• El efecto puede ser diferente"
    )

    print(
        "  para otras observaciones."
    )

    print(
        "\nConclusión:"
    )

    print(
        "El valor obtenido corresponde al"
    )

    print(
        "efecto causal estimado para una"
    )

    print(
        "unidad específica y no debe"
    )

    print(
        "interpretarse como el efecto"
    )

    print(
        "causal promedio de toda la población."
    )

except Exception as e:

    print(
        "No fue posible interpretar automáticamente "
        "el resultado."
    )

    print(e)

print(
    "FIN INTERPRETACIÓN DEL CATE CON TARGET DOUBLE MACHINE LEARNING\n\n\n\n"
)



# ATE DML + BOOTSTRAP
print(
    "CATE Object and Confidence Intervals"
)

dml_estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.econml.dml.DML",
    target_units="ate",
    confidence_intervals=True,
    method_params={
        "init_params": {
            "model_y":
                GradientBoostingRegressor(),

            "model_t":
                GradientBoostingRegressor(),

            "model_final":
                LassoCV(
                    fit_intercept=False
                ),

            "featurizer":
                PolynomialFeatures(
                    degree=1,
                    include_bias=True
                )
        },

        "fit_params": {
            "inference":
                BootstrapInference(
                    n_bootstrap_samples=100,
                    n_jobs=-1
                )
        }
    }
)

print(
    dml_estimate
)


print(
    "\n" + "=" * 70
)

print(
    "INTERPRETACIÓN DEL ATE (DOUBLE MACHINE LEARNING)"
)

print(
    "=" * 70
)

try:

    ate = dml_estimate.value

    print(
        f"\nATE estimado = {ate:.6f}"
    )

    print(
        "\n¿Qué significa este resultado?"
    )

    print(
        "---------------------------------------------"
    )

    print(
        "Se estimó el Average Treatment Effect (ATE)"
    )

    print(
        "utilizando Double Machine Learning (DML)."
    )

    print(
        "El ATE representa el cambio promedio esperado"
    )

    print(
        "en la variable resultado cuando el tratamiento"
    )

    print(
        "cambia del valor de control al valor tratado,"
    )

    print(
        "después de controlar los factores de confusión"
    )

    print(
        "identificados en el DAG."
    )

    if ate > 0:

        print(
            "\nInterpretación:"
        )

        print(
            "El tratamiento incrementa el resultado "
            f"en promedio {abs(ate):.4f} unidades."
        )

    elif ate < 0:

        print(
            "\nInterpretación:"
        )

        print(
            "El tratamiento reduce el resultado "
            f"en promedio {abs(ate):.4f} unidades."
        )

    else:

        print(
            "\nInterpretación:"
        )

        print(
            "No se observa un efecto causal promedio."
        )

    efecto = abs(ate)

    if efecto < 0.10:
        nivel = "muy pequeño"

    elif efecto < 0.50:
        nivel = "pequeño"

    elif efecto < 1:
        nivel = "moderado"

    elif efecto < 2:
        nivel = "grande"

    else:
        nivel = "muy grande"

    print(
        f"\nMagnitud del efecto: {nivel}"
    )

    try:

        ci = dml_estimate.get_confidence_intervals()

        li = float(ci[0][0])
        ls = float(ci[0][1])

        print(
            "\nIntervalo de confianza (Bootstrap)"
        )

        print(
            "---------------------------------------------"
        )

        print(
            f"Límite inferior : {li:.6f}"
        )

        print(
            f"Límite superior : {ls:.6f}"
        )

        if li <= 0 <= ls:

            print(
                "\nInterpretación del intervalo:"
            )

            print(
                "El intervalo de confianza incluye "
                "el valor 0."
            )

            print(
                "Por lo tanto, con el nivel de confianza"
            )

            print(
                "seleccionado no existe evidencia suficiente"
            )

            print(
                "para afirmar que el efecto causal promedio"
            )

            print(
                "es diferente de cero."
            )

        else:

            print(
                "\nInterpretación del intervalo:"
            )

            print(
                "El intervalo de confianza NO incluye "
                "el valor 0."
            )

            print(
                "Esto proporciona evidencia de que el efecto"
            )

            print(
                "causal promedio es estadísticamente diferente"
            )

            print(
                "de cero para el nivel de confianza utilizado."
            )

    except Exception:

        print(
            "\nNo fue posible recuperar los"
        )

        print(
            "intervalos de confianza."
        )

    print(
        "\n¿Por qué utilizar Double Machine Learning?"
    )

    print(
        "---------------------------------------------"
    )

    print(
        "• Modela el tratamiento mediante Machine Learning."
    )

    print(
        "• Modela el resultado mediante Machine Learning."
    )

    print(
        "• Reduce el sesgo por especificación incorrecta."
    )

    print(
        "• Captura relaciones no lineales."
    )

    print(
        "• Es robusto en problemas de alta dimensionalidad."
    )

    print(
        "\n¿Por qué utilizar Bootstrap?"
    )

    print(
        "---------------------------------------------"
    )

    print(
        "Bootstrap genera múltiples muestras"
    )

    print(
        "del conjunto de datos para cuantificar"
    )

    print(
        "la incertidumbre de la estimación."
    )

    print(
        "Esto permite construir intervalos"
    )

    print(
        "de confianza sin depender estrictamente"
    )

    print(
        "de supuestos de normalidad."
    )

    print(
        "\nConclusión:"
    )

    print(
        "El valor obtenido corresponde al efecto"
    )

    print(
        "causal promedio (ATE) estimado mediante"
    )

    print(
        "Double Machine Learning."
    )

    print(
        "Los intervalos de confianza obtenidos"
    )

    print(
        "mediante Bootstrap permiten evaluar la"
    )

    print(
        "precisión y la incertidumbre de la estimación."
    )

except Exception as e:

    print(
        "No fue posible interpretar automáticamente "
        "el resultado."
    )

    print(e)

print(
    "FIN INTERPRETACIÓN DEL ATE CON DOUBLE MACHINE LEARNING "
    "CON INTERVALOS DE CONFIANZA\n\n\n\n"
)



# CATE INDIVIDUALES: 20 OBSERVACIONES
print(
    "VARIANTES"
)

print(
    "Can provide a new inputs as target units and estimate CATE on them."
)

print(
    "Es posible proporcionar nuevas entradas como unidades "
    "objetivo y estimar el CATE para ellas."
)

effect_modifier_names = [
    'GRADUATION_YEAR',
    'SCORE_NEM',
    'AVERAGE_MATH_LANG',
    'GENDER',
    'HOW_MANY_WORK',
    'HOW_MANY_STUDY',
    'INCOME_PERCENTILE',
    'MOTHER_EDUCATION',
    'FATHER_EDUCATION',
    'PREVIOUS_ADMISSION',
    'SCHOOL_TYPE',
    'SCHOOL_DEPENDENCY',
    'WORK',
    'RURAL'
]

missing_modifiers = [
    c for c in effect_modifier_names
    if c not in data.columns
]

if missing_modifiers:

    raise Exception(
        f"Faltan variables del CSV para CATE: "
        f"{missing_modifiers}"
    )

n_test = min(
    20,
    len(data)
)

test_df = data[
    effect_modifier_names
].sample(
    n=n_test,
    random_state=123
)

print(
    test_df.head()
)

dml_estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.econml.dml.DML",
    target_units=test_df,
    confidence_intervals=False,
    method_params={
        "init_params": {
            "model_y":
                GradientBoostingRegressor(
                    random_state=123
                ),

            "model_t":
                GradientBoostingRegressor(
                    random_state=123
                ),

            "model_final":
                LassoCV(
                    cv=5,
                    random_state=123
                ),

            "featurizer":
                PolynomialFeatures(
                    degree=1,
                    include_bias=True
                )
        },

        "fit_params": {}
    }
)

print(
    dml_estimate.cate_estimates
)


print(
    "\n" + "=" * 70
)

print(
    "INTERPRETACIÓN DE LOS EFECTOS CAUSALES INDIVIDUALES (CATE)"
)

print(
    "=" * 70
)

try:

    cate = np.asarray(
        dml_estimate.cate_estimates
    ).flatten()

    print(
        f"\nNúmero de observaciones evaluadas : {len(cate)}"
    )

    print(
        "\n¿Qué significa este resultado?"
    )

    print(
        "---------------------------------------------"
    )

    print(
        "Se estimó un efecto causal individual"
    )

    print(
        "(Conditional Average Treatment Effect)"
    )

    print(
        "para cada observación contenida en"
    )

    print(
        "el conjunto de prueba (test_df)."
    )

    print(
        "\nCada valor representa cuánto cambia"
    )

    print(
        "la variable resultado para una"
    )

    print(
        "observación específica cuando pasa"
    )

    print(
        "del grupo de control al tratado."
    )

    print(
        "\nResumen de los efectos individuales"
    )

    print(
        "---------------------------------------------"
    )

    print(
        f"Promedio : {np.mean(cate):.4f}"
    )

    print(
        f"Desv. estándar : {np.std(cate):.4f}"
    )

    print(
        f"Mínimo : {np.min(cate):.4f}"
    )

    print(
        f"Máximo : {np.max(cate):.4f}"
    )

    print(
        f"Mediana : {np.median(cate):.4f}"
    )

    positivos = np.sum(
        cate > 0
    )

    negativos = np.sum(
        cate < 0
    )

    nulos = np.sum(
        cate == 0
    )

    print(
        "\nDistribución de efectos"
    )

    print(
        "---------------------------------------------"
    )

    print(
        f"Efectos positivos : {positivos}"
    )

    print(
        f"Efectos negativos : {negativos}"
    )

    print(
        f"Efectos nulos : {nulos}"
    )

    print(
        "\nPrimeros efectos individuales"
    )

    print(
        "---------------------------------------------"
    )

    for i, valor in enumerate(
        cate[:10]
    ):

        if valor > 0:
            sentido = "Aumenta"

        elif valor < 0:
            sentido = "Disminuye"

        else:
            sentido = "Sin efecto"

        print(
            f"Observación {i+1:3d}: "
            f"{valor:8.4f}   ({sentido})"
        )

    print(
        "\nInterpretación:"
    )

    if np.std(cate) < 0.05:

        print(
            "Los efectos causales son bastante"
        )

        print(
            "homogéneos entre los individuos."
        )

        print(
            "El tratamiento produce un efecto"
        )

        print(
            "similar para casi toda la población."
        )

    else:

        print(
            "Existe heterogeneidad en los efectos"
        )

        print(
            "causales."
        )

        print(
            "El impacto del tratamiento cambia"
        )

        print(
            "entre distintos individuos."
        )

    print(
        "\nConclusión:"
    )

    print(
        "Double Machine Learning permitió"
    )

    print(
        "estimar un efecto causal diferente"
    )

    print(
        "para cada observación del conjunto"
    )

    print(
        "de prueba."
    )

    print(
        "Estos resultados permiten identificar"
    )

    print(
        "qué individuos se benefician más"
    )

    print(
        "del tratamiento y cuáles presentan"
    )

    print(
        "efectos pequeños o incluso negativos."
    )

except Exception as e:

    print(
        "No fue posible interpretar automáticamente "
        "el resultado."
    )

    print(e)

print(
    "FIN INTERPRETACIÓN DE LOS CATE INDIVIDUALES \n\n\n\n"
)



# OBJETO ECONML
print(
    "Can also retrieve the raw EconML estimator object "
    "for any further operations"
)

print(
    "También es posible recuperar el objeto estimador "
    "EconML sin procesar para realizar operaciones adicionales."
)

try:

    print(
        dml_estimate._estimator_object
    )

except Exception as e:

    print(
        "No fue posible mostrar el objeto interno de EconML."
    )

    print(e)



# RESUMEN FINAL
print(
    "\n" + "=" * 70
)

print(
    "RESUMEN FINAL DEL ANALISIS PC + DoWhy"
)

print(
    "=" * 70
)

print(
    "\nCSV utilizado :", CSV_FILE
)

print(
    "DOT utilizado :", DOT_FILE
)

print(
    "Tratamiento   :", TREATMENT
)

print(
    "Resultado     :", TARGET
)

print(
    "\nNodos del DAG DoWhy:",
    graph.number_of_nodes()
)

print(
    "Aristas del DAG DoWhy:",
    graph.number_of_edges()
)

print(
    "Aristas descartadas por ciclos:",
    len(pc_rejected_edges)
)

print(
    "Aristas no orientadas del DOT:",
    len(pc_undirected_edges)
)

print(
    "\nANALISIS FINALIZADO."
)
