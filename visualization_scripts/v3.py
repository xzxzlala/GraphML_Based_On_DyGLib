import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

# === 数据准备 ===
df = pd.read_csv("./DG_data/mooc/ml_mooc.csv")

# 选择用户列表（reindexed 后的）
user_ids = [123, 456, 789]  # 替换为实际存在的 ID
num_users = len(user_ids)

# 时间切片
num_slices = 10
ts_edges = pd.cut(df['ts'], bins=num_slices, retbins=True)[1]

# 每个用户的时间帧邻居信息
user_frames_data = []
for uid in user_ids:
    user_data = []
    for i in range(len(ts_edges) - 1):
        slice_df = df[(df['ts'] >= ts_edges[i]) & (df['ts'] < ts_edges[i + 1])]
        user_df = slice_df[slice_df['u'] == uid]
        neighbors = set(user_df['i'].tolist())
        user_data.append(neighbors)
    user_frames_data.append(user_data)

# 每个用户的上一次邻居记录（用于判断新增/持续/消失）
prev_neighbors_list = [set() for _ in range(num_users)]

# === 绘图设置 ===
fig, axs = plt.subplots(1, num_users, figsize=(5 * num_users, 5))

def update(frame):
    for i, uid in enumerate(user_ids):
        ax = axs[i]
        ax.clear()
        current_neighbors = user_frames_data[i][frame]
        prev_neighbors = prev_neighbors_list[i]

        new_neighbors = current_neighbors - prev_neighbors
        persistent_neighbors = current_neighbors & prev_neighbors
        vanished_neighbors = prev_neighbors - current_neighbors

        G = nx.Graph()
        G.add_node(f"User {uid}")
        edge_colors = []
        edge_styles = []

        for item in persistent_neighbors:
            G.add_node(f"Item {item}")
            G.add_edge(f"User {uid}", f"Item {item}")
            edge_colors.append("blue")
            edge_styles.append("solid")

        for item in new_neighbors:
            G.add_node(f"Item {item}")
            G.add_edge(f"User {uid}", f"Item {item}")
            edge_colors.append("green")
            edge_styles.append("solid")

        for item in vanished_neighbors:
            G.add_node(f"Item {item}")
            G.add_edge(f"User {uid}", f"Item {item}")
            edge_colors.append("red")
            edge_styles.append("dashed")

        pos = nx.spring_layout(G, seed=42)
        for j, edge in enumerate(G.edges()):
            nx.draw_networkx_edges(G, pos, edgelist=[edge], ax=ax,
                                   edge_color=edge_colors[j],
                                   style=edge_styles[j], width=2)

        nx.draw_networkx_nodes(G, pos, node_size=400, ax=ax, node_color="#cccccc")
        nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)

        ax.set_title(f"User {uid} - Time {frame+1}/{num_slices}")
        ax.axis("off")

        # 添加图例（只在第一个子图中加即可）
        if i == 2:
            legend_elements = [
                mpatches.Patch(color='green', label='New'),
                mpatches.Patch(color='blue', label='Persistent'),
                mpatches.Patch(color='red', label='Vanished (dashed)'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        prev_neighbors_list[i] = current_neighbors.copy()

ani = FuncAnimation(fig, update, frames=num_slices, interval=1200)

# 保存为 gif（嵌入 PPT）
ani.save("multi_user_dynamic_with_legend.gif", writer="pillow")
plt.close()
