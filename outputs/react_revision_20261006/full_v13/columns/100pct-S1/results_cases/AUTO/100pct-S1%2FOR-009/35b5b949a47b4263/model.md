## Mathematical Model

Let:
- $I$ = set of areas (indexed by $i$), with area names as in the ProductName column of file_1_view_0.
- $x_i$ = scale of development per day in area $i$ (decision variable, nonnegative integer).
- $v_i$ = development benefit per unit in area $i$ (parameter, Value column in file_1_view_0).
- $w_i$ = development capacity required per unit in area $i$ (parameter, Weight column in file_1_view_0).
- $C$ = overall development capacity (parameter, Capacity column in file_0_view_0).

### Objective
\[
\max \sum_{i \in I} v_i x_i
\]

### Constraint
\[
\sum_{i \in I} w_i x_i \leq C
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

## Data Mapping

- $I$: All ProductName values in file_1_view_0 (products.csv), in original row order.
- $v_i$: Value column in file_1_view_0, matched to $i$ by ProductName.
- $w_i$: Weight column in file_1_view_0, matched to $i$ by ProductName.
- $C$: Capacity column in file_0_view_0 (capacity.csv), row 0.
- $x_i$: Decision variable for each $i \in I$.

All parameters and index sets are mapped directly from the returned CSV data, preserving file and column names.