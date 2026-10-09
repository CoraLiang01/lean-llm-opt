##### Mathematical Model

Let:
- $I$ = set of components (indexed by $i$), corresponding to all "Unnamed: 0" in file_1_view_0 (unit_price.csv) and columns in file_0_view_0 (processing_time_unit.csv).
- $K$ = set of workshops (indexed by $k$), corresponding to all "workshop" in file_2_view_0 (total_working_hours.csv) and rows in file_0_view_0 (processing_time_unit.csv).
- $x_i$ = number of units of component $i$ to produce (decision variable), $x_i \in \mathbb{Z}_{\geq 0}$.

Parameters:
- $p_i$ = unit price of component $i$ (from file_1_view_0, column "unit_price").
- $a_{ki}$ = unit processing time required for component $i$ in workshop $k$ (from file_0_view_0, row $k$, column $i$).
- $b_k$ = total available working hours in workshop $k$ (from file_2_view_0, column "total_hours").

Objective:
$$
\max \sum_{i \in I} p_i x_i
$$

Subject to (for all $k \in K$):
$$
\sum_{i \in I} a_{ki} x_i \leq b_k
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

##### Data Mapping

- $I$: All "Unnamed: 0" in file_1_view_0 (unit_price.csv) and all columns except "Unnamed: 0" in file_0_view_0 (processing_time_unit.csv).
- $K$: All "workshop" in file_2_view_0 (total_working_hours.csv) and all "Unnamed: 0" in file_0_view_0 (processing_time_unit.csv).
- $p_i$: file_1_view_0, column "unit_price", key "Unnamed: 0".
- $a_{ki}$: file_0_view_0, row "Unnamed: 0" = $k$, column $i$.
- $b_k$: file_2_view_0, column "total_hours", key "workshop".
- $x_i$: Decision variable, nonnegative integer, for each $i \in I$.