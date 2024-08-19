import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np

# 设置 numpy 的打印选项，以便打印完整的邻接矩阵
np.set_printoptions(threshold=np.inf)

# 创建一个空的图
G = nx.Graph()

# 添加节点
nodes = []
for i in range(1, 3):
    nodes.append(f'A{i}')
for i in range(3, 5):
    nodes.append(f'B{i}')
for i in range(5, 8):
    nodes.append(f'C{i}')
for i in range(8, 17):
    nodes.append(f'D{i}')
for i in range(17, 22):
    nodes.append(f'E{i}')
for i in range(22, 36):
    nodes.append(f'F{i}')
G.add_nodes_from(nodes)

# 生成随机位置
pos = {node: (random.randint(0, 500), random.randint(0, 500)) for node in nodes}

# 创建颜色列表
color_list = ["red", "red", "yellow", "yellow", "blue", "blue", "blue", "green", "green", "green", "green", "green",
              "green", "green", "green", "green", "purple", "purple", "purple", "purple", "purple", "orange", "orange",
              "orange", "orange", "orange", "orange", "orange", "orange", "orange", "orange", "orange", "orange",
              "orange", "orange"]

# 添加边
for i in range(1, 3):
    for j in range(3, 5):
        if random.random() <= 0.6 and G.degree[f'A{i}'] < 15 and G.degree[f'B{j}'] < 15:
            G.add_edge(f'A{i}', f'B{j}')
for i in range(1, 3):
    for j in range(5, 8):
        if random.random() <= 0.5 and G.degree[f'A{i}'] < 15 and G.degree[f'C{j}'] < 15:
            G.add_edge(f'A{i}', f'C{j}')
for i in range(1, 3):
    for j in range(8, 17):
        if random.random() <= 0.6 and G.degree[f'A{i}'] < 15 and G.degree[f'D{j}'] < 15:
            G.add_edge(f'A{i}', f'D{j}')
for i in range(1, 3):
    for j in range(17, 22):
        if random.random() <= 0.7 and G.degree[f'A{i}'] < 15 and G.degree[f'E{j}'] < 15:
            G.add_edge(f'A{i}', f'E{j}')
for i in range(1, 3):
    for j in range(22, 36):
        if random.random() <= 0.5 and G.degree[f'A{i}'] < 15 and G.degree[f'F{j}'] < 15:
            G.add_edge(f'A{i}', f'F{j}')
for i in range(3, 5):
    for j in range(5, 8):
        if random.random() <= 0.5 and G.degree[f'B{i}'] < 15 and G.degree[f'C{j}'] < 15:
            G.add_edge(f'B{i}', f'C{j}')
for i in range(8, 17):
    for j in range(17, 22):
        if random.random() <= 0.4 and G.degree[f'D{i}'] < 15 and G.degree[f'E{j}'] < 15:
            G.add_edge(f'D{i}', f'E{j}')
for i in range(8, 17):
    for j in range(22, 36):
        if i != j and random.random() <= 0.4 and G.degree[f'D{i}'] < 15 and G.degree[f'F{j}'] < 15:
            G.add_edge(f'D{i}', f'F{j}')
for i in range(17, 22):
    for j in range(22, 36):
        if i != j and random.random() <= 0.4 and G.degree[f'E{i}'] < 15 and G.degree[f'F{j}'] < 15:
            G.add_edge(f'E{i}', f'F{j}')
for i in range(5, 8):
    for j in range(5, 8):
        if i != j and random.random() < 0.3 and G.degree[f'C{i}'] < 15 and G.degree[f'C{j}'] < 15:
            G.add_edge(f'C{i}', f'C{j}')
for i in range(17, 22):
    for j in range(17, 22):
        if i != j and random.random() < 0.3 and G.degree[f'E{i}'] < 15 and G.degree[f'E{j}'] < 15:
            G.add_edge(f'E{i}', f'E{j}')
for i in range(22, 36):
    for j in range(22, 36):
        if i != j and random.random() < 0.3 and G.degree[f'F{i}'] < 15 and G.degree[f'F{j}'] < 15:
            G.add_edge(f'F{i}', f'F{j}')

# 绘制网络图
nx.draw(G, pos, with_labels=True, node_size=300, node_color=color_list, font_size=8, font_weight='bold',
        edge_color='gray')

# 设置坐标范围
plt.xlim(0, 500)
plt.ylim(0, 500)

# 显示坐标轴
plt.axis('on')

# 显示刻度
plt.xticks(range(0, 501, 100))
plt.yticks(range(0, 501, 100))

# 保存图形
plt.savefig('network_graph.png')

# 显示图形
plt.show()

# 将节点坐标导出到csv文件
df = pd.DataFrame(pos.items(), columns=['Node', 'Position'])
df['Type'] = df['Node'].apply(lambda x: x[0])
df.to_csv('node_positions.csv', index=False)

# 将边导出到csv文件
edges = list(G.edges())
edges_df = pd.DataFrame(edges, columns=['Node1', 'Node2'])
edges_df.to_csv('edges.csv', index=False)

# 计算6种节点的平均节点度
avg_degree = {}
for node_type in ['A', 'B', 'C', 'D', 'E', 'F']:
    nodes_of_type = [node for node in nodes if node.startswith(node_type)]
    avg_degree[node_type] = sum([G.degree[node] for node in nodes_of_type]) / len(nodes_of_type)

print('Average Degree:')
print(avg_degree)

# 绘制每个节点的度的直方图
plt.figure()
for node in nodes:
    plt.bar(node, G.degree[node])
plt.xlabel('Node')
plt.ylabel('Degree')
plt.savefig('degree_histogram.png')
plt.show()

# 绘制节点度的分布函数图
plt.figure()
degrees = [G.degree[node] for node in nodes]
degree_values, degree_counts = np.unique(degrees, return_counts=True)
degree_probs = degree_counts / len(nodes)
plt.bar(degree_values, degree_probs)
plt.xlabel('Degree')
plt.ylabel('Probability')
plt.savefig('degree_distribution.png')
plt.show()

# 生成邻接矩阵
adj_matrix = nx.adjacency_matrix(G)
print('Adjacency Matrix:')
print(adj_matrix.todense())

# 绘制邻接矩阵图像并保存
plt.figure()
plt.imshow(adj_matrix.todense(), cmap='binary', interpolation='none')
plt.colorbar(label='Edge')
plt.savefig('adjacency_matrix.png')
plt.show()

# 计算最短路径矩阵
shortest_path_matrix = np.zeros((len(G.nodes), len(G.nodes)))
for i, source in enumerate(G.nodes()):
    for j, target in enumerate(G.nodes()):
        shortest_path_matrix[i, j] = nx.shortest_path_length(G, source=source, target=target)
print('Shortest Path Matrix:')
print('节点的最短路径')

# 绘制最短路径矩阵图像并保存
plt.figure()
plt.imshow(shortest_path_matrix, cmap='viridis', interpolation='none')
plt.colorbar(label='Shortest Path Length')
plt.savefig('shortest_path_matrix.png')
plt.show()

# 计算平均最短路径长度
average_shortest_path = nx.average_shortest_path_length(G)
print("Average Shortest Path Length:", average_shortest_path)
