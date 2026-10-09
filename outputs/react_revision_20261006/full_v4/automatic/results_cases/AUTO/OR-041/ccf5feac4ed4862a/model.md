##### Mathematical Model

Let:
- $I$ = set of areas (indexed by $i$), as given by all ProductName in products.csv.
- $x_i$ = scale of development per day in area $i$ (decision variable, nonnegative integer).
- $v_i$ = development benefit of area $i$ (parameter).
- $w_i$ = development capacity required for area $i$ (parameter).
- $C$ = overall development capacity (parameter).

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Subject to:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

##### Data Mapping

- $I$: All ProductName in table_id file_1_view_0, column ProductName.
- $v_i$: file_1_view_0, column Value, for each $i$.
- $w_i$: file_1_view_0, column Weight, for each $i$.
- $C$: file_0_view_0, column Capacity.
- $x_i$: Decision variable for each $i \in I$.