## Mathematical Model

Sets:
- $I$: set of items, $I = \{\text{1}, \text{2}, \ldots, \text{140}\}$ (from value.csv, column "item")

Parameters (from value.csv, table_id: file_0_view_0):
- $v_i$: value of item $i$ ("value")
- $w_i$: weight of item $i$ ("weight")
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

- $I$: All "item" values in value.csv (table_id: file_0_view_0, column "item")
- $v_i$: value.csv (table_id: file_0_view_0, column "value", for each $i$)
- $w_i$: value.csv (table_id: file_0_view_0, column "weight", for each $i$)
- $W$: 15 (from user description)