import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from collections import defaultdict

# === 加载数据 ===
df = pd.read_csv("./DG_data/mooc/ml_mooc.csv")  # 替换成你的实际路径

# === 时间切片设置 ===
num_slices = 10
ts_edges = pd.cut(df['ts'], bins=num_slices, retbins=True)[1]

# === 每帧全图边集 ===
all_frames_edges = []
for i in range(len(ts_edges) - 1):
    slice_df = df[(df['ts'] >= ts_edges[i]) & (df['ts'] < ts_edges[i + 1])]
    edges = set([(f"u_{u}", f"i_{i}") for u, i in zip(slice_df['u'], slice_df['i'])])
    all_frames_edges.append(edges)

# === 统计最活跃节点（出现频率高）===
node_freq = defaultdict(int)
for edge_set in all_frames_edges:
    for u, v in edge_set:
        node_freq[u] += 1
        node_freq[v] += 1

subset_size = 15
top_nodes = sorted(node_freq, key=node_freq.get, reverse=True)[:subset_size]
best_subset = set(top_nodes)

# === 每帧筛选子图边集（只要一端在子集中）===
frames_edges = []
for edge_set in all_frames_edges:
    filtered = set([e for e in edge_set if e[0] in best_subset or e[1] in best_subset])
    frames_edges.append(filtered)

# === 检查是否有边被筛出来
print("每帧子图边数量:", [len(f) for f in frames_edges])

# === 动画绘图准备 ===
fig, ax = plt.subplots(figsize=(8, 8))
prev_edges = set()

def update(frame):
    global prev_edges
    ax.clear()
    current_edges = frames_edges[frame]

    # 限制边数量
    MAX_EDGES = 100
    current_edges = set(list(current_edges)[:MAX_EDGES])
    prev_edges = set(list(prev_edges)[:MAX_EDGES])

    G = nx.Graph()
    for e in current_edges | prev_edges:
        G.add_edge(*e)

    edge_colors = []
    edge_styles = []

    for e in current_edges:
        if e in prev_edges:
            edge_colors.append("blue")  # 持续边
            edge_styles.append("solid")
        else:
            edge_colors.append("green")  # 新增边
            edge_styles.append("solid")

    for e in prev_edges - current_edges:
        G.add_edge(*e)
        edge_colors.append("red")  # 消失边
        edge_styles.append("dashed")

    if len(G.nodes()) == 0:
        ax.set_title("⚠️ 子图为空")
        return

    pos = nx.spring_layout(G, seed=42)
    for i, edge in enumerate(G.edges()):
        nx.draw_networkx_edges(G, pos, edgelist=[edge], ax=ax,
                               edge_color=edge_colors[i],
                               style=edge_styles[i], width=2)

    nx.draw_networkx_nodes(G, pos, node_color="#bbbbbb", node_size=300, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)

    # 添加图例 legend
    legend_elements = [
        mpatches.Patch(color='green', label='New Edge'),
        mpatches.Patch(color='blue', label='Persistent Edge'),
        mpatches.Patch(color='red', label='Vanished Edge (dashed)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    ax.set_title(f"Dynamic Subgraph (Top {subset_size} Nodes) — Time Slice {frame+1}/{num_slices}")
    ax.axis("off")

    prev_edges.clear()
    prev_edges.update(current_edges)

ani = FuncAnimation(fig, update, frames=num_slices, interval=1200)

# === 保存为 gif 动画（适合插入PPT）===
ani.save("mooc_dynamic_top_subgraph_final.gif", writer="pillow")
plt.close()
