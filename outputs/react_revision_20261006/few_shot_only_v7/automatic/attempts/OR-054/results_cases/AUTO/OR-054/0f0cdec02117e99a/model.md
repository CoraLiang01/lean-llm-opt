##### Mathematical Model

Let:
- $S$ = set of shelves, indexed by $i$ (from capacity.csv, column ShelfID)
- $P$ = set of products, indexed by $j$ (from products.csv, column ProductName)

Parameters:
- $c_i$ = capacity of shelf $i$ (capacity.csv, Capacity)
- $v_j$ = value per unit of product $j$ (products.csv, Value)
- $w_j$ = weight (space requirement) per unit of product $j$ (products.csv, Weight)

Decision variables:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on shelf $i$

Objective:
$$
\max \sum_{i \in S} \sum_{j \in P} v_j x_{ij}
$$

Subject to:
1. Shelf capacity constraints (for all $i \in S$):
$$
\sum_{j \in P} w_j x_{ij} \leq c_i
$$

2. Nonnegativity and integrality (for all $i \in S$, $j \in P$):
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

##### Data Mapping

- $S$ (shelves): /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/capacity.csv, column ShelfID
- $P$ (products): /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/products.csv, column ProductName
- $c_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/capacity.csv, column Capacity, keyed by ShelfID
- $v_j$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/products.csv, column Value, keyed by ProductName
- $w_j$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/products.csv, column Weight, keyed by ProductName