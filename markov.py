import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'


# Определяем состояния
states = ["S0", "S1", "S2", "S3"]

# Определяем вероятности переходов (ребра графа)
transitions = {
    ("S0", "S1"): 0.3,
    ("S0", "S2"): 0.7,
    ("S1", "S3"): 0.4,
    ("S1", "S0"): 0.6,
    ("S2", "S3"): 1.0,
    ("S3", "S0"): 1.0,
}

# Создаем граф
G = nx.DiGraph()

# Добавляем рёбра с весами
for (start, end), prob in transitions.items():
    G.add_edge(start, end, weight=prob)

# Рисуем граф
pos = nx.spring_layout(G)  # Определяем расположение узлов
plt.figure(figsize=(6, 6))
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=12)
edge_labels = {(start, end): f"{prob:.2f}" for (start, end), prob in transitions.items()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
plt.title("Граф переходов Марковского процесса")
plt.show()
