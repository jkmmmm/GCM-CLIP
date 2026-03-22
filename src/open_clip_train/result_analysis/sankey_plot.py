import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# 读取数据
df = pd.read_csv('matches_summary.csv')

# 1. 构建 disease -> cluster 的关系
disease_cluster = df.groupby(['disease', 'cluster_id']).size().reset_index(name='count')
disease_cluster['cluster_id'] = 'Cluster ' + disease_cluster['cluster_id'].astype(str)

# 2. 构建 location -> cluster 的关系
location_cluster = df.groupby(['location', 'cluster_id']).size().reset_index(name='count')
location_cluster['cluster_id'] = 'Cluster ' + location_cluster['cluster_id'].astype(str)

# 3. 创建桑基图数据
def create_sankey_data(df, source_col, target_col, value_col):
    # 获取所有节点
    nodes = list(df[source_col].unique()) + list(df[target_col].unique())
    node_dict = {node: i for i, node in enumerate(nodes)}
    
    # 构建连接
    sources = [node_dict[src] for src in df[source_col]]
    targets = [node_dict[tgt] for tgt in df[target_col]]
    values = df[value_col].tolist()
    
    return nodes, sources, targets, values

# 创建两个桑基图
fig1 = go.Figure()
fig2 = go.Figure()

# 第一个桑基图：disease -> cluster
nodes1, sources1, targets1, values1 = create_sankey_data(
    disease_cluster, 'disease', 'cluster_id', 'count'
)

fig1.add_trace(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=nodes1,
        color="blue"
    ),
    link=dict(
        source=sources1,
        target=targets1,
        value=values1,
        color="rgba(100, 149, 237, 0.6)"
    )
))

fig1.update_layout(
    title_text="Disease 到 Cluster 的分布关系",
    font_size=12,
    height=700
)

# 第二个桑基图：location -> cluster
nodes2, sources2, targets2, values2 = create_sankey_data(
    location_cluster, 'location', 'cluster_id', 'count'
)

fig2.add_trace(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=nodes2,
        color="green"
    ),
    link=dict(
        source=sources2,
        target=targets2,
        value=values2,
        color="rgba(144, 238, 144, 0.6)"
    )
))

fig2.update_layout(
    title_text="Location 到 Cluster 的分布关系",
    font_size=12,
    height=700
)

# 显示图表
fig1.show()
fig2.show()

# 保存为 PNG 文件（需要安装 kaleido）
fig1.write_image("disease_cluster_sankey.png", width=1600, height=900, scale=2)
fig2.write_image("location_cluster_sankey.png", width=1600, height=900, scale=2)

print("PNG文件已保存：disease_cluster_sankey.png, location_cluster_sankey.png")