##### Mathematical Model

Let $I$ be the set of shelves (from file_0_view_0, column ShelfID), and $J$ be the set of products (from file_1_view_0, column ProductName).

Let $x_{ij} \geq 0$ denote the number of units of product $j \in J$ placed on shelf $i \in I$ (continuous or integer, as not specified).

Parameters:
- $c_i$: capacity of shelf $i$ (file_0_view_0, column Capacity)
- $v_j$: value per unit of product $j$ (file_1_view_0, column Value)
- $w_j$: weight per unit of product $j$ (file_1_view_0, column Weight)

Objective:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
$$

Subject to:
$$
\sum_{j \in J} w_j x_{ij} \leq c_i \quad \forall i \in I
$$
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

##### Data Mapping

- $I$ (Shelves): file_0_view_0, column ShelfID
- $c_i$: file_0_view_0, column Capacity, keyed by ShelfID
- $J$ (Products): file_1_view_0, column ProductName
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName
- $x_{ij}$: decision variable, number of units of product $j$ on shelf $i$ (continuous, nonnegative)

All sets and parameters are defined by the full, unsimplified contents of the respective columns in the returned tables.