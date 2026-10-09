##### Mathematical Model

Let:
- $I$ = set of drug products (indexed by $i$), with identifiers from file_1_view_0.ProductName
- For each $i \in I$:
    - $v_i$ = benefit per unit of drug $i$ (file_1_view_0.Value)
    - $w_i$ = weight per unit of drug $i$ (file_1_view_0.Weight)
- $C$ = overall stock capacity (file_0_view_0.Capacity)
- $x_i$ = number of units of drug $i$ to order each day (decision variable, integer, $x_i \geq 0$)

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

##### Data Mapping

- $I$: file_1_view_0.ProductName
- $v_i$: file_1_view_0.Value, for each $i$
- $w_i$: file_1_view_0.Weight, for each $i$
- $C$: file_0_view_0.Capacity
- $x_i$: decision variable for each $i \in I$