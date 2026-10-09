Mathematical Model

Index Sets:
Let $I$ be the set of bread types, with each $i \in I$ corresponding to a unique ProductName from file_1_view_0.

Parameters:
For each $i \in I$:
- $p_i$: expected profit per unit of bread $i$ (Value from file_1_view_0, column Value)
- $w_i$: storage weight per unit of bread $i$ (Weight from file_1_view_0, column Weight)

Let $C$ be the total storage capacity (Capacity from file_0_view_0, column Capacity).

Decision Variables:
For each $i \in I$:
- $x_i$: number of units of bread $i$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

Objective:
$\max \sum_{i \in I} p_i x_i$

Constraint:
$\sum_{i \in I} w_i x_i \leq C$

Variable Domains:
$x_i \in \mathbb{Z}_{\geq 0}$ for all $i \in I$

Data Mapping

Index Sets:
- $I$: All ProductName values in file_1_view_0

Parameters:
- $p_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity

Decision Variables:
- $x_i$: number of units to order of bread $i$ (indexed by ProductName from file_1_view_0)

All parameters and index sets are mapped directly from the specified columns and rows in the returned CSVQA data.