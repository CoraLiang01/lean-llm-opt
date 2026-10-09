## Mathematical Model

**Sets**
- $I$: set of items, $I = \{\text{1}, \text{2}, \ldots, \text{140}\}$ (from column "item" in value.csv)

**Parameters** (from value.csv, table_id: file_0_view_0)
- $v_i$: value of item $i$ (column "value")
- $w_i$: weight of item $i$ (column "weight")
- $W = 15$: total weight capacity

**Decision Variables**
- $x_i \in \{0,1\}$: 1 if item $i$ is selected, 0 otherwise, for all $i \in I$

**Objective**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraints**
\[
\sum_{i \in I} w_i x_i \leq W
\]
\[
x_i \in \{0,1\} \quad \forall i \in I
\]

---

### Data Mapping

- $I$: All rows in value.csv, column "item", table_id: file_0_view_0
- $v_i$: value.csv, column "value", table_id: file_0_view_0, for each $i$
- $w_i$: value.csv, column "weight", table_id: file_0_view_0, for each $i$
- $W$: 15 (from problem statement)