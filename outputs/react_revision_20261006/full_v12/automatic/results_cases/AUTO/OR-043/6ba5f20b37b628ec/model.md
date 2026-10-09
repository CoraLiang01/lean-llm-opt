ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: set of drug products, indexed by $i$ (from file_1_view_0, column ProductName)

Parameters:
- $v_i$: benefit per unit of drug $i$ (file_1_view_0, column Value, key ProductName)
- $w_i$: weight per unit of drug $i$ (file_1_view_0, column Weight, key ProductName)
- $C$: overall stock capacity (file_0_view_0, column Capacity)

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: number of units of drug $i$ to order each day

Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \geq 0,\quad x_i \in \mathbb{Z},\quad \forall i \in I
\]

Data Mapping:
- $I$: file_1_view_0, column ProductName
- $v_i$: file_1_view_0, column Value, key ProductName
- $w_i$: file_1_view_0, column Weight, key ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: number of units of drug $i$ to order each day (decision variable, indexed by ProductName)