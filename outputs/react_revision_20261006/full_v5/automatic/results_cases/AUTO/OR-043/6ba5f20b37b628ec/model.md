#### Mathematical Model

Let:
- $I$ = set of drug products (indexed by $i$), with identifiers ProductName from file_1_view_0.
- For each $i \in I$:
    - $v_i$ = Value of product $i$ (from column Value in file_1_view_0)
    - $w_i$ = Weight of product $i$ (from column Weight in file_1_view_0)
- $C$ = overall stock capacity (from Capacity in file_0_view_0)
- $x_i$ = number of units of product $i$ to order each day (decision variable, nonnegative integer)

Objective:
$$
\max \sum_{i \in I} v_i x_i
$$

Subject to:
$$
\sum_{i \in I} w_i x_i \leq C
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

#### Data Mapping

- $I$: All ProductName in file_1_view_0 (products.csv)
- $v_i$: Value column in file_1_view_0 (products.csv), keyed by ProductName
- $w_i$: Weight column in file_1_view_0 (products.csv), keyed by ProductName
- $C$: Capacity column in file_0_view_0 (capacity.csv)
- $x_i$: Decision variable for each $i \in I$ (nonnegative integer)