#### Sets
- $I$: set of areas (indexed by $i$), corresponding to all ProductName values in file_1_view_0.

#### Parameters
- $b_i$: benefit coefficient for area $i$ (from Value in file_1_view_0, keyed by ProductName).
- $w_i$: development unit weight for area $i$ (from Weight in file_1_view_0, keyed by ProductName).
- $C$: overall development capacity (from Capacity in file_0_view_0, single value).

#### Decision Variables
- $x_i \in \mathbb{Z}_{\geq 0}$: integer scale of development in area $i$ per day.

#### Objective
\[
\max \sum_{i \in I} b_i x_i
\]

#### Constraints
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

#### Data Mapping

- $I$: All ProductName values in file_1_view_0 (products.csv), in original row order.
- $b_i$: file_1_view_0, column Value, keyed by ProductName.
- $w_i$: file_1_view_0, column Weight, keyed by ProductName.
- $C$: file_0_view_0, column Capacity, single value.
- $x_i$: Decision variable for each $i \in I$.

No additional constraints or parameters are present in the data. All mappings use the exact column names and file/table IDs as returned.