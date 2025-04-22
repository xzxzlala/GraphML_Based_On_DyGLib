import pandas as pd
from collections import Counter

df = pd.read_csv("./DG_data/mooc/ml_mooc.csv")

# 你指定的核心用户
core_users = [123, 7090]

# 找出所有和他们连接过的 item 节点
linked_items = []
for uid in core_users:
    linked_items += df[df['u'] == uid]['i'].tolist()

# 找频率最高的 item
item_counter = Counter(linked_items)
top_items = [item for item, _ in item_counter.most_common(10)]

# 再找出除了 core_users 以外，最活跃的其他用户（避免空图）
other_users = df['u'].value_counts().index.tolist()
other_users = [u for u in other_users if u not in core_users][:8]

# 构建完整固定节点集
fixed_ids = core_users + top_items + other_users  # 123 + 7090 + 10 item + 8 other user = 20

print("✅ 选中的固定节点 ID（数字形式）:", fixed_ids)

# 构建 best_subset，加上 u_/i_ 前缀
best_subset = set()
for x in fixed_ids:
    if x in df['u'].values:
        best_subset.add(f"u_{x}")
    elif x in df['i'].values:
        best_subset.add(f"i_{x}")

print("✅ best_subset（用于图中显示的点）:", best_subset)
