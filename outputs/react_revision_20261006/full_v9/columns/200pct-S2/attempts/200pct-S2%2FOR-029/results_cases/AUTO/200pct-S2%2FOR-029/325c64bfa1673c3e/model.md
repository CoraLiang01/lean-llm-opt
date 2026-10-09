ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: set of displays (from file_0_view_0.ShelfID)
- $J$: set of products (from file_1_view_0.ProductName)

Parameters:
- $c_i$: capacity of display $i$ (from file_0_view_0, column Capacity, key ShelfID)
- $v_j$: value of product $j$ (from file_1_view_0, column Value, key ProductName)
- $w_j$: weight of product $j$ (from file_1_view_0, column Weight, key ProductName)

Decision Variables:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on display $i$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to:
1. Display capacity constraints (for all $i \in I$):
\[
\sum_{j \in J} w_j x_{ij} \leq c_i
\]
2. Minimum total quantity of the first product ("Smartphone") across all displays:
\[
\sum_{i \in I} x_{i,\,\text{Smartphone}} \geq 5
\]
3. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

DATA MAPPING

- $I$: file_0_view_0.ShelfID
- $J$: file_1_view_0.ProductName
- $c_i$: file_0_view_0.Capacity, keyed by ShelfID
- $v_j$: file_1_view_0.Value, keyed by ProductName
- $w_j$: file_1_view_0.Weight, keyed by ProductName
- "Smartphone": file_1_view_1.ProductName (first product for total quantity constraint)
- $x_{ij}$: allocation of product $j$ to display $i$ (decision variable, nonnegative integer)