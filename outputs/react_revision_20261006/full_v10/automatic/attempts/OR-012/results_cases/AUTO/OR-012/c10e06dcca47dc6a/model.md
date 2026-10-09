Mathematical Optimization Model

Index Sets:
- $I$: set of all products, indexed by $i$ (from all "Product Name" entries in file_0_view_0)

Parameters:
- $A_i$: revenue per unit of product $i$ ("Revenue", file_0_view_0)
- $d_i$: deterministic demand for product $i$ ("Demand", file_0_view_0)
- $I_i$: initial inventory for product $i$ ("Initial Inventory", file_0_view_0)

Decision Variables:
- $x_i$: number of units of product $i$ to fulfill for customer purchases, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$

Objective:
\[
\max \sum_{i \in I} A_i x_i
\]

Constraints:
1. Inventory constraint:
\[
x_i \leq I_i \quad \forall i \in I
\]
2. Demand constraint:
\[
x_i \leq d_i \quad \forall i \in I
\]
3. Non-negativity and integrality:
\[
x_i \in \mathbb{Z}_+, \quad \forall i \in I
\]

Data Mapping:
- Table: file_0_view_0 (OnlineSalesDataset.csv)
    - Index set $I$: all values in "Product Name"
    - Parameter $A_i$: "Revenue"
    - Parameter $d_i$: "Demand"
    - Parameter $I_i$: "Initial Inventory"