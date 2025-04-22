import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

# === 加载数据 ===
df = pd.read_csv("./DG_data/mooc/ml_mooc.csv")

# === 时间切片 ===
num_slices = 10
ts_edges = pd.cut(df['ts'], bins=num_slices, retbins=True)[1]

# === 构建每帧全图边集 ===
all_frames_edges = []
for i in range(len(ts_edges) - 1):
    slice_df = df[(df['ts'] >= ts_edges[i]) & (df['ts'] < ts_edges[i + 1])]
    edges = set([(f"u_{u}", f"i_{i}") for u, i in zip(slice_df['u'], slice_df['i'])])
    all_frames_edges.append(edges)

# ✅ 固定的 20 个点（你在 v5_select.py 中已选好）
best_subset = {
    'u_123', 'u_1182', 'u_1687', 'u_2991', 'u_806',
    'u_2359', 'u_2059', 'u_233', 'u_3271',
    'i_7090', 'i_7078', 'i_7084', 'i_7089', 'i_7049',
    'i_7082', 'i_7080', 'i_7051', 'i_7061', 'i_7057', 'i_7050'
}

# === 每帧只保留子集内部边
frames_edges = []
for edge_set in all_frames_edges:
    filtered = set([e for e in edge_set if e[0] in best_subset and e[1] in best_subset])
    frames_edges.append(filtered)

# === 绘图准备
fig, ax = plt.subplots(figsize=(8, 8))
prev_edges = set()
# === 创建一个完整图 G_full 用于固定布局
G_full = nx.Graph()
for edge_set in frames_edges:
    for e in edge_set:
        G_full.add_edge(*e)

# 固定 layout（只生成一次）
pos = nx.spring_layout(G_full, seed=42)

def update(frame):
    global prev_edges
    ax.clear()
    current_edges = frames_edges[frame]

    G = nx.Graph()
    for e in current_edges | prev_edges:
        G.add_edge(*e)

    edge_colors = []
    edge_styles = []

    for e in current_edges:
        if e in prev_edges:
            edge_colors.append("blue")  # 持续
            edge_styles.append("solid")
        else:
            edge_colors.append("green")  # 新增
            edge_styles.append("solid")

    for e in prev_edges - current_edges:
        G.add_edge(*e)
        edge_colors.append("red")
        edge_styles.append("dashed")

    if len(G.edges()) == 0:
        ax.set_title(f"⛔ No edges in frame {frame+1}")
        return

    #pos = nx.spring_layout(G, seed=42)
    for i, edge in enumerate(G.edges()):
        nx.draw_networkx_edges(G, pos, edgelist=[edge], ax=ax,
                               edge_color=edge_colors[i],
                               style=edge_styles[i], width=2)

    nx.draw_networkx_nodes(G, pos, node_color="#cccccc", node_size=300, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)

    legend_elements = [
        mpatches.Patch(color='green', label='New Edge'),
        mpatches.Patch(color='blue', label='Persistent Edge'),
        mpatches.Patch(color='red', label='Vanished Edge (dashed)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    ax.set_title(f"Dynamic of Fixed Subgraph (20 Nodes) — Time Slice {frame+1}/{num_slices}")
    ax.axis("off")

    prev_edges.clear()
    prev_edges.update(current_edges)

ani = FuncAnimation(fig, update, frames=num_slices, interval=1200)
ani.save("mooc_fixed_20nodes.gif", writer="pillow")
plt.close()
