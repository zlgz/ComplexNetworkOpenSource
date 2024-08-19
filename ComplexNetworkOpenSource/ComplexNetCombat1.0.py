import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
from matplotlib.pyplot import MultipleLocator


def setup():
    np.set_printoptions(threshold=np.inf)


# 创建复杂随机网络图
def create_graph():
    G = nx.Graph()
    nodes = create_nodes()
    G.add_nodes_from(nodes)
    pos = generate_positions(nodes)
    color_list = assign_colors(nodes)
    add_edges(G)
    return G, pos, color_list


# 随机创建6种类型的35个节点
def create_nodes():
    prefix_ranges = [
        ('C', (1, 3)),
        ('M1', (3, 5)),
        ('M2', (5, 8)),
        ('U1', (8, 17)),
        ('U2', (17, 22)),
        ('U3', (22, 36))
    ]
    nodes = [f'{prefix}{i}' for prefix, ranges in prefix_ranges for i in range(*ranges)]
    return nodes


# 在500×500的范围内随机生成位置
def generate_positions(nodes):
    return {node: (random.randint(0, 500), random.randint(0, 500)) for node in nodes}


# 配置颜色
def assign_colors(nodes):
    color_map = {'C': 'red', 'M1': 'yellow', 'M2': 'blue', 'U1': 'green', 'U2': 'purple', 'U3': 'orange'}
    # 确保前缀能正确匹配颜色
    return [color_map[node[:2]] if node.startswith(('M1', 'M2', 'U1', 'U2', 'U3')) else color_map[node[0]] for node in nodes]


# 添加边
def add_edges(G):
    nodes = list(G.nodes())
    for i, node in enumerate(nodes):
        for j, target in enumerate(nodes):
            if i != j and random_edge(node, target, G):
                G.add_edge(node, target)

    # 确保图的连通性，如果不是连通的，则添加额外的边来连接不同的连通分量
    ensure_connectivity(G)


# 确保图的连通性
def ensure_connectivity(G):
    # 获取所有连通分量
    components = list(nx.connected_components(G))
    connected_nodes = set()

    # 遍历每个连通分量，将其与之前的连通分量相连接
    for component in components:
        if connected_nodes:
            # 选择当前组件中的一个节点和已连接节点集中的一个节点
            current_node = next(iter(component))
            connected_node = next(iter(connected_nodes))
            # 添加一条边来连接这两个连通分量
            if not G.has_edge(current_node, connected_node):
                G.add_edge(current_node, connected_node)
        connected_nodes.update(component)


# 按照给定的概率随机生成边
def random_edge(node, target, G):
    # 用于确定节点类型的辅助函数
    def get_node_type(node):
        if node.startswith(('M1', 'M2', 'U1', 'U2', 'U3')):
            return node[:2]  # 返回前两个字符作为节点类型
        return node[0]  # 否则返回第一个字符作为节点类型

    node_type = get_node_type(node)
    target_type = get_node_type(target)

    # 度限制条件
    if G.degree[node] >= 15 or G.degree[target] >= 15:
        return False

    # 节点间连接概率
    probability_map = {
        ('C', 'M1'): 0.5, ('C', 'M2'): 0.5, ('C', 'U1'): 0.6,
        ('C', 'U2'): 0.6, ('C', 'U3'): 0.6, ('M1', 'M2'): 0.5,
        ('U1', 'U2'): 0.3, ('U1', 'U3'): 0.3, ('U2', 'U3'): 0.3,
        ('M1', 'M1'): 0.4, ('M2', 'M2'): 0.4,  # 注意这里修正了原有的映射，以匹配 'M1' 和 'M2'
        ('U1', 'U1'): 0.3, ('U2', 'U2'): 0.3, ('U3', 'U3'): 0.3
    }

    # 检查节点类型组合是否在概率映射中，如果在，使用给定的连接概率
    if (node_type, target_type) in probability_map:
        return random.random() <= probability_map[(node_type, target_type)]
    elif (target_type, node_type) in probability_map:  # 反向组合
        return random.random() <= probability_map[(target_type, node_type)]

    # 默认不连接
    return False


# 注意: 该函数逻辑是基于你之前的代码设计的。实际使用时，需要确保它被正确集成到add_edges函数中，并且考虑了节点间不能自连接的规则（即 if node != target: ...）。


# 绘制创建的网络图并保存，可选择是否绘制边
def draw_and_save_graph(G, pos, color_list, draw_edges=True):
    plt.figure(figsize=(10, 8), dpi=100)
    ax = plt.gca()

    # 根据节点的度来调整节点大小`
    # node_sizes = [G.degree(node) * 30 for node in G.nodes()]  # 每个节点的度乘以100作为大小
    if draw_edges:
        nx.draw(G, pos, ax=ax, node_color=color_list, node_size=100, with_labels=True, edge_color="gray")
    else:
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=color_list)
        nx.draw_networkx_labels(G, pos, ax=ax)
    ax.axis('on')  # 关闭坐标轴
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    # 设置坐标轴的范围，以确保原点在左下角。假设您希望坐标轴的范围根据节点的位置自动调整
    x_values, y_values = zip(*pos.values())
    ax.set_xlim(min(x_values) - 50, max(x_values) + 50)  # 给定一定的边缘空间
    ax.set_ylim(min(y_values) - 10, max(y_values) + 50)  # 给定一定的边缘空间

    # 开启坐标轴的网格
    # ax.grid(color='gray', linestyle='--', linewidth=0.2)
    filename = 'network_graph_with_edges.png' if draw_edges else 'network_graph_without_edges.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


# 导出创建的边数据和节点数据保存到csv文件
def export_data(G, pos):
    export_node_positions(pos, 'node_positions.csv')
    export_edges(G, 'edges.csv')


# 导出边数据
def export_edges(G, filename):
    edges_df = pd.DataFrame(list(G.edges()), columns=['Node1', 'Node2'])
    edges_df.to_csv(filename, index=False)


# 导出节点位置数据
def export_node_positions(pos, filename):
    df = pd.DataFrame(list(pos.items()), columns=['Node', 'Position'])
    df.to_csv(filename, index=False)


# 分析创建的复杂网络图
def analyze_graph(G):
    avg_degree = calculate_average_degree(G)
    print('Average Degree:', avg_degree)
    draw_degree_histogram(G)
    draw_degree_distribution(G)
    print_adjacency_matrix(G)
    calculate_and_show_shortest_paths(G)


# 计算节点平均度
# 计算节点平均度
def calculate_average_degree(G):
    # 初始化一个字典来存储每种类型的度总和和节点数量
    degree_sums = {'C': 0, 'M1': 0, 'M2': 0, 'U1': 0, 'U2': 0, 'U3': 0}
    node_counts = {'C': 0, 'M1': 0, 'M2': 0, 'U1': 0, 'U2': 0, 'U3': 0}

    # 遍历所有节点，根据节点类型累加度数并计数
    for node in G.nodes():
        node_type = node[:2] if node.startswith(('M1', 'M2', 'U1', 'U2', 'U3')) else node[0]
        degree_sums[node_type] += G.degree(node)
        node_counts[node_type] += 1

    # 计算并返回每种类型的平均度
    avg_degree = {node_type: (degree_sums[node_type] / node_counts[node_type] if node_counts[node_type] > 0 else 0)
                  for node_type in degree_sums}

    return avg_degree


# 绘制节点度的直方图
def draw_degree_histogram(G):
    plt.figure()
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 1, 1), alpha=0.75)
    plt.title('Degree Histogram')
    plt.xlabel('Degree')
    plt.ylabel('Number of Nodes')
    plt.savefig('degree_histogram.png')
    plt.show()


# 绘制度分布函数图
def draw_degree_distribution(G):
    degrees = [G.degree(n) for n in G.nodes()]
    degree_values, degree_counts = np.unique(degrees, return_counts=True)
    plt.figure()
    plt.bar(degree_values, degree_counts / sum(degree_counts))
    plt.title('Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Fraction of Nodes')
    plt.savefig('degree_distribution.png')
    plt.show()


# 输出网络图的邻接矩阵
def print_adjacency_matrix(G):
    adj_matrix = nx.adjacency_matrix(G).todense()
    print('Adjacency Matrix:\n', adj_matrix)


# 计算并显示最短路径
def calculate_and_show_shortest_paths(G):
    # This can be very heavy for large graphs; might want to skip for very large G
    path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    num_nodes = len(G.nodes())
    shortest_path_matrix = np.zeros((num_nodes, num_nodes))

    # 填充最短路径矩阵
    for i, source_node in enumerate(G.nodes()):
        for j, target_node in enumerate(G.nodes()):
            if target_node in path_lengths[source_node]:
                shortest_path_matrix[i, j] = path_lengths[source_node][target_node]
            else:
                shortest_path_matrix[i, j] = np.inf  # 如果没有路径，则设为无穷大

    # 显示最短路径长度矩阵的可视化
    plt.figure(figsize=(10, 10))
    plt.imshow(shortest_path_matrix, cmap='viridis', interpolation='none')
    plt.colorbar(label='Shortest Path Length')
    plt.xticks(range(num_nodes), list(G.nodes()), rotation=90)
    plt.yticks(range(num_nodes), list(G.nodes()))
    plt.tight_layout()
    plt.savefig('shortest_path_matrix.png')
    plt.show()

    avg_path_length = nx.average_shortest_path_length(G)
    print("Average Shortest Path Length:", avg_path_length)

    # 输出最短路径矩阵
    print("Shortest Path Matrix:\n", shortest_path_matrix)


# 绘制节点度的条形图
def draw_degree_bar_chart(G):
    plt.figure(figsize=(12, 6))
    degrees = [G.degree[node] for node in sorted(G.nodes(), key=lambda x: int(x[1:]))]
    nodes = [node for node in sorted(G.nodes(), key=lambda x: int(x[1:]))]
    plt.bar(range(1, 36), degrees, tick_label=nodes)
    plt.xlabel('Node')
    plt.ylabel('Degree')
    plt.title('Node Degree')
    plt.xticks(rotation=90)  # 旋转x轴标签，避免重叠
    plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
    plt.savefig('degree_bar_chart.png')
    plt.show()


# 节点度的直方图
def node_degree_zhifangtu(G):
    plt.figure(figsize=(12.8, 12.8))
    for node in G.nodes:
        plt.bar(node, G.degree[node])
    plt.xticks(rotation=45)
    x_major_locator = MultipleLocator(2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlabel('Node')
    plt.ylabel('Degree')
    plt.savefig('degree_histogram.png')
    plt.show()


# 绘制聚类系数直方图和平均聚类系数
def draw_clustering_coefficient_histogram_and_average(G):
    # 计算聚类系数
    clustering_coeffs = nx.clustering(G)
    # 计算每种类型的平均聚类系数
    types = ['C', 'M1', 'M2', 'U1', 'U2', 'U3']
    avg_clustering_coeffs = {}
    for t in types:
        coeffs = [clustering_coeffs[node] for node in G.nodes if node.startswith(t)]
        avg_clustering_coeffs[t] = np.mean(coeffs) if coeffs else 0

    # 绘制直方图
    plt.figure(figsize=(12, 8))
    nodes, coeffs = zip(*sorted(clustering_coeffs.items(), key=lambda x: x[0]))
    plt.bar(nodes, coeffs, label='Clustering Coefficient')

    # 绘制每种类型的平均聚类系数折线图
    averages = [avg_clustering_coeffs[node[:2]] if node.startswith(('M1', 'M2', 'U1', 'U2', 'U3')) else avg_clustering_coeffs[node[0]] for node in nodes]
    plt.plot(nodes, averages, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=6, label='Average Clustering Coefficient')

    plt.xlabel('Node')
    plt.ylabel('Clustering Coefficient')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig('clustering_coefficient_histogram.png')
    plt.show()

    # 打印每种类型的平均聚类系数
    for t, avg_coeff in avg_clustering_coeffs.items():
        print(f"Average Clustering Coefficient for {t}: {avg_coeff}")


# 绘制介数中心性直方图
def draw_betweenness_centrality_histogram_and_average(G):
    # 计算介数中心性
    betweenness_centrality = nx.betweenness_centrality(G)

    # 计算每种类型的平均介数中心性
    types = ['C', 'M1', 'M2', 'U1', 'U2', 'U3']
    avg_betweenness = {}
    for t in types:
        # 对于每种类型，筛选出该类型的节点，计算它们的介数中心性平均值
        centrality_values = [betweenness_centrality[node] for node in G.nodes if node.startswith(t)]
        avg_betweenness[t] = np.mean(centrality_values) if centrality_values else 0

    # 绘制介数中心性直方图
    plt.figure(figsize=(12, 8))
    nodes, centrality_values = zip(*sorted(betweenness_centrality.items(), key=lambda x: x[0]))
    plt.bar(nodes, centrality_values, color='green', alpha=0.75)

    plt.xlabel('Node')
    plt.ylabel('Betweenness Centrality')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('betweenness_centrality_histogram.png')
    plt.show()

    # 输出每种类型的平均介数中心性
    for t, avg in avg_betweenness.items():
        print(f"Average Betweenness Centrality for {t}: {avg}")


# 主函数
def main():
    setup()
    G, pos, color_list = create_graph()
    draw_and_save_graph(G, pos, color_list, draw_edges=False)
    draw_and_save_graph(G, pos, color_list,draw_edges=True)
    draw_degree_bar_chart(G)
    export_data(G, pos)
    analyze_graph(G)
    node_degree_zhifangtu(G)
    draw_clustering_coefficient_histogram_and_average(G)
    draw_betweenness_centrality_histogram_and_average(G)


if __name__ == "__main__":
    main()
