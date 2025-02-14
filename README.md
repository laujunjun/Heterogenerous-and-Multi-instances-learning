# Heterogenerous-and-Multi-instances-learning

## 1. 异构图

### 1.1 简介

基于Heterogeneous Graph Attention Network(HAN)与Multi-Instance Learning(MIL)的乳腺癌生存预测任务。

#### 1.1.1 HAN的介绍

HAN（异质图注意力网络）是一种针对**异质图**（Heterogeneous Graphs）的深度学习模型，主要用于处理具有不同类型节点和边的图数据。

它通过**注意力机制（Attention Mechanism**来学习不同类型节点和关系的重要性，并进行特征聚合，适用于结构复杂的数据，如社交网络、生物网络和多模态数据分析。

HAN 主要基于**图注意力网络**（GAT, Graph Attention Network），但专门针对异质图进行了优化，主要包含以下部分：

(1) 节点级注意力（Node-level Attention）

- 由于不同类型的节点可能具有不同的影响力，HAN 使用**注意力机制**（Attention Mechanism）计算某个节点对其邻居节点的重要性。它使用**自注意力机制**（Self-attention）计算权重，使得对当前节点贡献更大的邻居具有更高的权重。

(2) 关系级注意力（Relation-level Attention）

- 在异质图中，不同类型的关系（边）对于模型的重要性不同。例如，在社交网络中，“朋友关系” 可能比 “点赞关系” 更重要。HAN 通过**关系级注意力**计算不同关系的重要性，并对所有关系进行加权聚合。

(3) 计算过程
	- 基于不同类型邻居的 GAT 计算节点嵌入（Node-level Attention）。
	- 针对不同关系（边）的注意力计算重要性（Relation-level Attention）。
	- 融合所有关系信息，形成最终的节点表示。

#### 1.1.2 MIL的介绍
**多实例学习**（MIL）是一种**弱监督学习**（Weakly Supervised Learning）方法，最初用于解决目标检测和医学影像分析等问题。

传统的机器学习任务需要每个样本都有明确的标签，但MIL只提供**包**（Bag）级别的标签，而每个包内的**实例**（Instance）是否具有相同的标签是未知的。

MIL的核心方法

- MIL 的关键是如何从多个 **实例**（Instances） 中 **选出最重要的实例** 并用于分类。常见的方法包括：

(1) Max Pooling MIL
	- 选择 最重要的实例（例如一个癌变区域），仅用这个实例进行决策。
	- 优点：计算简单，适合极端情况。
	- 缺点：可能忽略 Bag 中的其他关键信息，导致信息损失。

(2) Attention-based MIL（基于注意力的 MIL）
	- 使用 注意力机制 计算每个实例对整个 Bag 的贡献。
	- 例如，在病理图像中，可能有 多个癌变区域，不是单个区域决定结果。
	- 通过学习不同实例的重要性权重，更精确地建模整个 Bag 的特征。

### 1.2 实现

#### 1.2.1 图数据构建

代码：get_dataset.py

**（1）读取数据**
- get_f(path, filename) 读取 图节点 和 边信息：
  - CSV 文件：
  	- Edges.csv（边信息）
  	- Feats_T.csv（T类节点）
    - Feats_S.csv（S类节点）
  	- Feats_I.csv（I类节点）
  - 解析 Centroid 字段，提取 (x, y) 坐标信息。
- get_node(node_t, node_s, node_i, box)：
  - 通过 Bounding Box 选取某个区域内的节点。

**(2) 生成图数据**
- get_graph(edge, node_t, node_s, node_i, y, window_bbox)：
  - 通过 get_edge() 计算 同类型和跨类型的边索引。
	- 创建 HeteroData：
  	- 赋予 feature（23 维输入特征）
  	- 赋予 edge_index
  	- T.ToUndirected() 转换为无向图

**(3)数据加载**
  - get_dataset(df, feature_dir)：
    - 遍历 train.csv 和 val.csv，加载每个 病理图像 的特征。
  	- DataLoader 进行批量加载。

#### 1.2.2 HAN模型
代码：han.py
```
class HAN(torch.nn.Module):
    def __init__(self,  input_size, hidden_channels,  heads, pooling_ratio):
```

- HANConv 进行异质图特征学习
- TopKPooling 选择关键节点
- global_mean_pool 聚合特征
- Linear+BatchNorm 进行特征映射



#### 1.2.3 MIL
代码：MIL.py
```
class MIL_fc(torch.nn.Module):
    def __init__(self, size_arg = "small", dropout = 0.5, n_classes = 2, top_k=1,
                 embed_dim = 64, batch_size = 1):
```
- 通过 HAN 提取特征
- top_k 选择最具判别性的 patch
- classifier 进行二分类
- softmax 计算分类概率


#### 1.2.4注意力机制
```
class Attn_Net(nn.Module):
```
- Attn_Net：
  - 采用 两层 MLP
  - Tanh 进行非线性变换
- Attn_Net_Gated：
  - 采用 门控注意力
  - Sigmoid & Tanh 结合
