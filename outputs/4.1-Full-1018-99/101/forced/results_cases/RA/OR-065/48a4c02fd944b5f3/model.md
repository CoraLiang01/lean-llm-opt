Minimize
$$
\sum_{i \in \{\text{S1},\text{S2},\text{S3}\}} \text{fixed\_costs}_i \cdot y_i + \sum_{i \in \{\text{S1},\text{S2},\text{S3}\}} \sum_{j \in \{\text{C1},\text{C2},\text{C3}\}} \text{transport\_cost}_{ij} \cdot x_{ij}
$$

Subject to

1. Demand satisfaction for each customer:
$$
\sum_{i \in \{\text{S1},\text{S2},\text{S3}\}} x_{ij} = \text{demand}_j \qquad \forall j \in \{\text{C1},\text{C2},\text{C3}\}
$$

2. Warehouse activation constraints:
$$
x_{ij} \leq \text{demand}_j \cdot y_i \qquad \forall i \in \{\text{S1},\text{S2},\text{S3}\},\ \forall j \in \{\text{C1},\text{C2},\text{C3}\}
$$

3. Variable domains:
$$
y_i \in \{0,1\} \qquad \forall i \in \{\text{S1},\text{S2},\text{S3}\}
$$
$$
x_{ij} \geq 0 \qquad \forall i \in \{\text{S1},\text{S2},\text{S3}\},\ \forall j \in \{\text{C1},\text{C2},\text{C3}\}
$$

Where:

- Warehouses: S1, S2, S3
- Customers: C1, C2, C3

Parameters:

- Fixed costs:
  - $\text{fixed\_costs}_{\text{S1}} = 102.33$
  - $\text{fixed\_costs}_{\text{S2}} = 94.92$
  - $\text{fixed\_costs}_{\text{S3}} = 91.83$

- Demands:
  - $\text{demand}_{\text{C1}} = 1083$
  - $\text{demand}_{\text{C2}} = 776$
  - $\text{demand}_{\text{C3}} = 16214$

- Transportation costs:
  - $\text{transport\_cost}_{\text{S1},\text{C1}} = 1506.22$
  - $\text{transport\_cost}_{\text{S1},\text{C2}} = 70.90$
  - $\text{transport\_cost}_{\text{S1},\text{C3}} = 8.44$
  - $\text{transport\_cost}_{\text{S2},\text{C1}} = 1732.65$
  - $\text{transport\_cost}_{\text{S2},\text{C2}} = 1780.72$
  - $\text{transport\_cost}_{\text{S2},\text{C3}} = 567.44$
  - $\text{transport\_cost}_{\text{S3},\text{C1}} = 115.66$
  - $\text{transport\_cost}_{\text{S3},\text{C2}} = 100.76$
  - $\text{transport\_cost}_{\text{S3},\text{C3}} = 64.68$

Decision variables:

- $y_i$: binary, 1 if warehouse $i$ is open, 0 otherwise
- $x_{ij}$: quantity supplied from warehouse $i$ to customer $j$ (continuous, $\geq 0$)