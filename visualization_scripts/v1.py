import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import gridspec

# 加载数据（MOOC边列表）
df = pd.read_csv("./DG_data/mooc/ml_mooc.csv")

# 选择一个用户 ID（注意，已经被 reindex，从1开始）
user_id = 123

# 时间段划分
time_bins = ['early', 'mid', 'late']
df['period'] = pd.qcut(df['ts'], q=3, labels=time_bins)

# 创建子图
fig = plt.figure(figsize=(12, 4))
gs = gridspec.GridSpec(1, 3)

# 存储每段的邻接信息
neighbors_by_period = {}

for idx, period in enumerate(time_bins):
    period_df = df[(df['period'] == period) & (df['u'] == user_id)]
    neighbors = set(period_df['i'].tolist())
    neighbors_by_period[period] = neighbors

    # 构建子图
    ax = fig.add_subplot(gs[0, idx])
    G = nx.Graph()
    user_node = f"User {user_id}"
    G.add_node(user_node)

    item_nodes = [f"Item {item}" for item in neighbors]
    G.add_nodes_from(item_nodes)
    for item_node in item_nodes:
        G.add_edge(user_node, item_node)

    pos = nx.spring_layout(G, seed=42)

    # 颜色区分
    node_colors = []
    for node in G.nodes():
        if node == user_node:
            node_colors.append("#cccccc")  # 灰色 = 用户
        else:
            node_colors.append("#76c7c0")  # 绿色 = 课程

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    ax.set_title(f"{period.capitalize()} Period")

plt.suptitle("User-Item Interactions over Time (MOOC)")
plt.tight_layout()
plt.show()
