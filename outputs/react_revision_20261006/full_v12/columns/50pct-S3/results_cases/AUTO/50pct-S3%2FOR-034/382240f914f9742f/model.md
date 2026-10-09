## Mathematical Model

Sets:
- $I$: set of items, $I = \{\text{1}, \text{2}, \ldots, \text{140}\}$ (from column "item" in value.csv, table_id: file_0_view_0)

Parameters (from value.csv, table_id: file_0_view_0):
- $v_i$: value of item $i$, from column "value"
- $w_i$: weight of item $i$, from column "weight"
- $W = 15$: total weight capacity

Decision variables:
- $x_i \in \{0,1\}$: 1 if item $i$ is selected, 0 otherwise

Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq W
\]
\[
x_i \in \{0,1\} \quad \forall i \in I
\]

---

### Data Mapping

- $I$: All rows in value.csv, table_id: file_0_view_0, column "item"
- $v_i$: value.csv, table_id: file_0_view_0, column "value"
- $w_i$: value.csv, table_id: file_0_view_0, column "weight"
- $W$: 15 (from user description)
- $x_i$: binary decision variable for each $i \in I$