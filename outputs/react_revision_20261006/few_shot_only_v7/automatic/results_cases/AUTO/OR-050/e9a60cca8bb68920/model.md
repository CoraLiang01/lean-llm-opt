##### Sets
- $I$: set of displays (indexed by $i$), from capacity.csv, column ShelfID
- $J$: set of products (indexed by $j$), from products.csv, column ProductName

##### Parameters
- $c_i$: capacity of display $i$ (from capacity.csv, column Capacity)
- $v_j$: value per unit of product $j$ (from products.csv, column Value)
- $w_j$: weight per unit of product $j$ (from products.csv, column Weight)

##### Decision Variables
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on display $i$

##### Objective
Maximize total value:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
$$

##### Constraints

1. **Display Capacity Constraints** (for each display $i$):
$$
\sum_{j \in J} w_j x_{ij} \leq c_i \quad \forall i \in I
$$

2. **Minimum Quantity for First Product** (let $j^*$ be the first product in products.csv):
$$
\sum_{i \in I} x_{i j^*} \geq 5
$$

3. **Nonnegativity and Integrality**:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
$$

---

#### Data Mapping

- $I$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/capacity.csv, column ShelfID
- $c_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/capacity.csv, column Capacity
- $J$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/products.csv, column ProductName
- $v_j$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/products.csv, column Value
- $w_j$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/products.csv, column Weight
- $j^*$: first record in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/products.csv, column ProductName