ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: set of drug products, indexed by $i$ (from file_1_view_0, column ProductName)

Parameters:
- $v_i$: benefit per unit of drug $i$ (from file_1_view_0, column Value)
- $w_i$: weight (stock space required) per unit of drug $i$ (from file_1_view_0, column Weight)
- $C$: overall stock capacity (from file_0_view_0, column Capacity)

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
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

Data Mapping:
- $I$: All records in file_1_view_0 (products.csv), column ProductName
- $v_i$: file_1_view_0 (products.csv), column Value, keyed by ProductName
- $w_i$: file_1_view_0 (products.csv), column Weight, keyed by ProductName
- $C$: file_0_view_0 (capacity.csv), column Capacity
- $x_i$: decision variable for each $i \in I$ (drug product)