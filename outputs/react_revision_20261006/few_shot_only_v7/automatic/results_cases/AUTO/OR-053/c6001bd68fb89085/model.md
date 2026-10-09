Mathematical Optimization Model

Sets:
- $S$: set of shelves (indexed by $s$), from capacity.csv, column ShelfID
- $P$: set of products (indexed by $p$), from products.csv, column ProductName

Parameters:
- $C_s$: capacity of shelf $s$ (capacity.csv, column Capacity)
- $v_p$: value of product $p$ (products.csv, column Value)
- $w_p$: weight of product $p$ (products.csv, column Weight)

Decision Variables:
- $x_{sp}$: number of units of product $p$ placed on shelf $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

Subject to:
1. Shelf capacity constraints (for all $s \in S$):
\[
\sum_{p \in P} w_p \cdot x_{sp} \leq C_s
\]
2. Integrality and nonnegativity (for all $s \in S$, $p \in P$):
\[
x_{sp} \in \mathbb{Z}_{\geq 0}
\]

Data Mapping

- $S$ (shelves): /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/capacity.csv, column ShelfID
- $C_s$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/capacity.csv, column Capacity
- $P$ (products): /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv, column ProductName
- $v_p$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv, column Value
- $w_p$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv, column Weight