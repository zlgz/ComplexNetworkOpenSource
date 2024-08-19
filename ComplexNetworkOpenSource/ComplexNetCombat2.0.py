# -*- coding:utf-8 -*-
# @FileName  :Complex.py
# @Time      :2024/8/16 15:41
# @Author    :wuzefei
# @WeChat Official Accounts     :扎了个扎

import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
from matplotlib.pyplot import MultipleLocator
import matplotlib as mpl

# 配置matplotlib以使用LaTeX进行渲染
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['pgf.texsystem'] = 'pdflatex'
mpl.rcParams['font.size'] = 18  # 设置全局字体大小
mpl.rcParams['axes.labelsize'] = 18  # 坐标轴标签的字体大小
mpl.rcParams['xtick.labelsize'] = 14  # x轴刻度标签的字体大小
mpl.rcParams['ytick.labelsize'] = 14  # y轴刻度标签的字体大小
mpl.rcParams['legend.fontsize'] = 16  # 图例的字体大小
mpl.rcParams['figure.titlesize'] = 20  # 图形标题的字体大小


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


# 随机创建6种类型的35个节点，并用LaTeX格式表示
def create_nodes():
    prefix_ranges = [
        ('C', (1, 3)),
        ('M_{1}', (3, 5)),
        ('M_{2}', (5, 8)),
        ('U_{1}', (8, 17)),
        ('U_{2}', (17, 22)),
        ('U_{3}', (22, 36))
    ]
    nodes = [f'{prefix}^{{{i}}}' for prefix, ranges in prefix_ranges for i in range(*ranges)]
    return nodes


# 在500×500的范围内随机生成位置
def generate_positions(nodes):
    return {node: (random.randint(0, 500), random.randint(0, 500)) for node in nodes}


# 配置颜色
def assign_colors(nodes):
    color_map = {'C': 'red', 'M_{1}': 'yellow', 'M_{2}': 'blue', 'U_{1}': 'green', 'U_{2}': 'purple', 'U_{3}': 'orange'}
    return [color_map[node.split('^')[0]] for node in nodes]


# 添加边
def add_edges(G):
    nodes = list(G.nodes())
    for i, node in enumerate(nodes):
        for j, target in enumerate(nodes):
            if i != j and random_edge(node, target, G):
                G.add_edge(node, target)

    ensure_connectivity(G)


# 确保图的连通性
def ensure_connectivity(G):
    components = list(nx.connected_components(G))
    connected_nodes = set()

    for component in components:
        if connected_nodes:
            current_node = next(iter(component))
            connected_node = next(iter(connected_nodes))
            if not G.has_edge(current_node, connected_node):
                G.add_edge(current_node, connected_node)
        connected_nodes.update(component)


# 按照给定的概率随机生成边
def random_edge(node, target, G):
    def get_node_type(node):
        return node.split('^')[0]

    node_type = get_node_type(node)
    target_type = get_node_type(target)

    if G.degree[node] >= 15 or G.degree[target] >= 15:
        return False

    probability_map = {
        ('C', 'M_{1}'): 0.5, ('C', 'M_{2}'): 0.5, ('C', 'U_{1}'): 0.6,
        ('C', 'U_{2}'): 0.6, ('C', 'U_{3}'): 0.6, ('M_{1}', 'M_{2}'): 0.5,
        ('U_{1}', 'U_{2}'): 0.3, ('U_{1}', 'U_{3}'): 0.3, ('U_{2}', 'U_{3}'): 0.3,
        ('M_{1}', 'M_{1}'): 0.4, ('M_{2}', 'M_{2}'): 0.4,
        ('U_{1}', 'U_{1}'): 0.3, ('U_{2}', 'U_{2}'): 0.3, ('U_{3}', 'U_{3}'): 0.3
    }

    if (node_type, target_type) in probability_map:
        return random.random() <= probability_map[(node_type, target_type)]
    elif (target_type, node_type) in probability_map:
        return random.random() <= probability_map[(target_type, node_type)]

    return False


# 绘制创建的网络图并保存
def draw_and_save_graph(G, pos, color_list, draw_edges=True):
    plt.figure(figsize=(10, 10), dpi=100)
    ax = plt.gca()

    # 使用LaTeX渲染节点标签
    labels = {node: f"${{\Large {node}}}$" for node in G.nodes()}

    if draw_edges:
        nx.draw(G, pos, ax=ax, node_color=color_list, node_size=180, with_labels=True, edge_color="gray", labels=labels)
    else:
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=color_list)
        nx.draw_networkx_labels(G, pos, ax=ax, labels=labels)
    ax.axis('on')
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    x_values, y_values = zip(*pos.values())
    ax.set_xlim(min(x_values) - 50, max(x_values) + 50)
    ax.set_ylim(min(y_values) - 10, max(y_values) + 50)

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
def calculate_average_degree(G):
    degree_sums = {'C': 0, 'M_{1}': 0, 'M_{2}': 0, 'U_{1}': 0, 'U_{2}': 0, 'U_{3}': 0}
    node_counts = {'C': 0, 'M_{1}': 0, 'M_{2}': 0, 'U_{1}': 0, 'U_{2}': 0, 'U_{3}': 0}

    for node in G.nodes():
        node_type = node.split('^')[0]
        degree_sums[node_type] += G.degree(node)
        node_counts[node_type] += 1

    avg_degree = {node_type: (degree_sums[node_type] / node_counts[node_type] if node_counts[node_type] > 0 else 0)
                  for node_type in degree_sums}

    return avg_degree


# 绘制节点度的直方图
def draw_degree_histogram(G):
    plt.figure(figsize=(10, 8), dpi=100)
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 1, 1), alpha=0.75)
    plt.title(r'Degree Histogram')  # 使用r''确保LaTeX渲染
    plt.xlabel(r'Degree')
    plt.ylabel(r'Number of Nodes')
    plt.savefig('degree_histogram.png')
    plt.show()


# 绘制度分布函数图
def draw_degree_distribution(G):
    degrees = [G.degree(n) for n in G.nodes()]
    degree_values, degree_counts = np.unique(degrees, return_counts=True)
    plt.figure(figsize=(10, 8), dpi=100)
    plt.bar(degree_values, degree_counts / sum(degree_counts))
    plt.title(r'Degree Distribution')
    plt.xlabel(r'Degree')
    plt.ylabel(r'Fraction of Nodes')
    plt.savefig('degree_distribution.png')
    plt.show()


# 输出网络图的邻接矩阵
def print_adjacency_matrix(G):
    adj_matrix = nx.adjacency_matrix(G).todense()
    print('Adjacency Matrix:\n', adj_matrix)


# 计算并显示最短路径
# 计算并显示最短路径
def calculate_and_show_shortest_paths(G):
    path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    num_nodes = len(G.nodes())
    shortest_path_matrix = np.zeros((num_nodes, num_nodes))

    # 填充最短路径矩阵
    for i, source_node in enumerate(G.nodes()):
        for j, target_node in enumerate(G.nodes()):
            if target_node in path_lengths[source_node]:
                shortest_path_matrix[i, j] = path_lengths[source_node][target_node]
            else:
                shortest_path_matrix[i, j] = np.inf

    # 计算并打印平均最短路径长度
    avg_path_length = nx.average_shortest_path_length(G)
    print("Average Shortest Path Length:", avg_path_length)

    # 显示最短路径矩阵的可视化
    plt.figure(figsize=(10, 10), dpi=100)
    plt.imshow(shortest_path_matrix, cmap='viridis', interpolation='none')
    plt.colorbar(label=r'Shortest Path Length')

    # 使用 LaTeX 数学模式格式化节点标签
    labels = [f"${node}$" for node in G.nodes()]
    plt.xticks(range(num_nodes), labels, rotation=90)
    plt.yticks(range(num_nodes), labels)

    plt.tight_layout()
    plt.savefig('shortest_path_matrix.png')
    plt.show()


# 绘制节点度的条形图
def draw_degree_bar_chart(G):
    plt.figure(figsize=(10, 8), dpi=100)
    degrees = [G.degree[node] for node in sorted(G.nodes(), key=lambda x: int(x.split('^')[1].strip('{}')))]
    nodes = [node for node in sorted(G.nodes(), key=lambda x: int(x.split('^')[1].strip('{}')))]
    plt.bar(range(1, 36), degrees, tick_label=[f"${node}$" for node in nodes])
    plt.xlabel(r'Node')
    plt.ylabel(r'Degree')
    plt.title(r'Node Degree')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('degree_bar_chart.png')
    plt.show()


# 节点度的直方图
def node_degree_zhifangtu(G):
    plt.figure(figsize=(10, 8), dpi=100)

    # 获取所有节点的标签和度数
    nodes = list(G.nodes)
    degrees = [G.degree(node) for node in nodes]

    # 使用 LaTeX 数学模式格式化节点标签
    labels = [f"${node}$" for node in nodes]

    # 一次性绘制所有节点的条形图
    plt.bar(labels, degrees)

    # 设置 x 轴标签的旋转角度
    plt.xticks(rotation=45)

    # 设置 x 轴主刻度间隔
    x_major_locator = MultipleLocator(2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    # 设置坐标轴标签
    plt.xlabel(r'Node')
    plt.ylabel(r'Degree')

    # 确保布局紧凑
    plt.tight_layout()

    # 保存图像
    plt.savefig(r'degree_histogram.png')
    plt.show()


# 绘制聚类系数直方图和平均聚类系数
def draw_clustering_coefficient_histogram_and_average(G):
    clustering_coeffs = nx.clustering(G)
    types = ['C', 'M_{1}', 'M_{2}', 'U_{1}', 'U_{2}', 'U_{3}']
    avg_clustering_coeffs = {}
    for t in types:
        coeffs = [clustering_coeffs[node] for node in G.nodes if node.startswith(t)]
        avg_clustering_coeffs[t] = np.mean(coeffs) if coeffs else 0

    plt.figure(figsize=(20, 8), dpi=100)
    nodes, coeffs = zip(*sorted(clustering_coeffs.items(), key=lambda x: x[0]))
    plt.bar([f"${node}$" for node in nodes], coeffs, label=r'Clustering Coefficient')

    averages = [avg_clustering_coeffs[node.split('^')[0]] for node in nodes]
    plt.plot([f"${node}$" for node in nodes], averages, color='red', marker='o', linestyle='dashed', linewidth=2,
             markersize=6, label=r'Average Clustering Coefficient')

    plt.xlabel(r'Node')
    plt.ylabel(r'Clustering Coefficient')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig('clustering_coefficient_histogram.png')
    plt.show()

    for t, avg_coeff in avg_clustering_coeffs.items():
        print(f"Average Clustering Coefficient for {t}: {avg_coeff}")


# 绘制介数中心性直方图
def draw_betweenness_centrality_histogram_and_average(G):
    betweenness_centrality = nx.betweenness_centrality(G)

    types = ['C', 'M_{1}', 'M_{2}', 'U_{1}', 'U_{2}', 'U_{3}']
    avg_betweenness = {}
    for t in types:
        centrality_values = [betweenness_centrality[node] for node in G.nodes if node.startswith(t)]
        avg_betweenness[t] = np.mean(centrality_values) if centrality_values else 0

    plt.figure(figsize=(20, 8), dpi=100)
    nodes, centrality_values = zip(*sorted(betweenness_centrality.items(), key=lambda x: x[0]))
    plt.bar([f"${node}$" for node in nodes], centrality_values, color='green', alpha=0.75)

    plt.xlabel(r'Node')
    plt.ylabel(r'Betweenness Centrality')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('betweenness_centrality_histogram.png')
    plt.show()

    for t, avg in avg_betweenness.items():
        print(f"Average Betweenness Centrality for {t}: {avg}")


# 主函数
# 主函数
def main():
    setup()
    G, pos, color_list = create_graph()
    draw_and_save_graph(G, pos, color_list, draw_edges=False)
    draw_and_save_graph(G, pos, color_list, draw_edges=True)
    draw_degree_bar_chart(G)
    export_data(G, pos)
    analyze_graph(G)
    node_degree_zhifangtu(G)
    draw_clustering_coefficient_histogram_and_average(G)
    draw_betweenness_centrality_histogram_and_average(G)


if __name__ == "__main__":
    main()
