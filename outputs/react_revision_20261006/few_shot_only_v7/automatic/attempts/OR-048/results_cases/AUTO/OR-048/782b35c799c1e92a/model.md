##### Mathematical Model

Let:
- $I$ = set of storage areas (indexed by $i$), corresponding to all StorageID in capacity.csv
- $J$ = set of air conditioner types (indexed by $j$), corresponding to all ProductName in products.csv

Parameters:
- $c_i$ = capacity of storage area $i$ (from capacity.csv, column Capacity)
- $v_j$ = value of air conditioner type $j$ (from products.csv, column Value)
- $w_j$ = weight (size) of air conditioner type $j$ (from products.csv, column Weight)

Decision variables:
- $x_{ij}$ = number of units of air conditioner type $j$ placed in storage area $i$ ($x_{ij} \in \mathbb{Z}_{\geq 0}$)

Objective:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$

Subject to:
1. Storage area capacity constraints:
$$
\sum_{j \in J} w_j \cdot x_{ij} \leq c_i \quad \forall i \in I
$$

2. Nonnegativity and integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
$$

---

##### Data Mapping

- $I$: All StorageID in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/capacity.csv, column StorageID
- $c_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/capacity.csv, column Capacity, keyed by StorageID
- $J$: All ProductName in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv, column ProductName
- $v_j$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv, column Value, keyed by ProductName
- $w_j$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv, column Weight, keyed by ProductName
- $x_{ij}$: Number of units of product $j$ in storage area $i$, integer, $\geq 0$