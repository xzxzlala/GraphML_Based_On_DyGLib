import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# === 读取数据 ===
df = pd.read_csv("./DG_data/mooc/ml_mooc.csv")
user_id = 123  # 你可以换成任意 reindexed 的 user

# === 时间切片 ===
num_slices = 10
ts_edges = pd.cut(df['ts'], bins=num_slices, retbins=True)[1]

# === 每帧邻居集合 ===
frames_data = []
for i in range(len(ts_edges) - 1):
    slice_df = df[(df['ts'] >= ts_edges[i]) & (df['ts'] < ts_edges[i + 1])]
    user_df = slice_df[slice_df['u'] == user_id]
    neighbors = set(user_df['i'].tolist())
    frames_data.append(neighbors)

# === 动画准备 ===
fig, ax = plt.subplots(figsize=(6, 6))

# 记录历史邻居
prev_neighbors = set()

def update(frame):
    global prev_neighbors
    ax.clear()

    G = nx.Graph()
    G.add_node(f"User {user_id}")

    current_neighbors = frames_data[frame]

    # 分类邻居
    new_neighbors = current_neighbors - prev_neighbors
    persistent_neighbors = current_neighbors & prev_neighbors
    vanished_neighbors = prev_neighbors - current_neighbors

    # 加边
    edge_colors = []
    for item in persistent_neighbors:
        G.add_node(f"Item {item}")
        G.add_edge(f"User {user_id}", f"Item {item}")
        edge_colors.append("blue")

    for item in new_neighbors:
        G.add_node(f"Item {item}")
        G.add_edge(f"User {user_id}", f"Item {item}")
        edge_colors.append("green")

    # 消失的邻居以红色虚线标出（仅闪现）
    for item in vanished_neighbors:
        G.add_node(f"Item {item}")
        G.add_edge(f"User {user_id}", f"Item {item}")
        edge_colors.append("red")

    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=500, font_size=8, ax=ax,
            edge_color=edge_colors, style=['solid' if c != 'red' else 'dashed' for c in edge_colors])
    
    ax.set_title(f"Time Slice {frame+1}/{num_slices}")
    prev_neighbors = current_neighbors.copy()

ani = FuncAnimation(fig, update, frames=len(frames_data), interval=1200)

# === 保存动画 ===
ani.save("mooc_user_interaction_colored.gif", writer="pillow")
plt.close()
