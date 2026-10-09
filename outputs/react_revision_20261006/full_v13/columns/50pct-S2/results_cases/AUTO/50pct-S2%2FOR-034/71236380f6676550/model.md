## Symbolic Mathematical Model

**Sets**  
Let $I$ be the set of items, as indexed by the column "item" in value.csv:  
$I = \{\text{1}, \text{2}, \ldots, \text{140}\}$

**Parameters**  
For each $i \in I$:
- $v_i$: value of item $i$ (from column "value" in value.csv, table_id: file_0_view_0)
- $w_i$: weight of item $i$ (from column "weight" in value.csv, table_id: file_0_view_0)
- $W = 15$: total weight capacity

**Decision Variables**  
For each $i \in I$:
- $x_i \in \{0,1\}$: 1 if item $i$ is selected for display, 0 otherwise

**Objective**  
Maximize total value:
$$
\max \sum_{i \in I} v_i x_i
$$

**Constraints**  
Total weight does not exceed the counter limit:
$$
\sum_{i \in I} w_i x_i \leq W
$$

Binary selection:
$$
x_i \in \{0,1\} \quad \forall i \in I
$$

---

## Data Mapping

- $I$: All "item" values in value.csv (table_id: file_0_view_0)
- $v_i$: "value" column in value.csv (table_id: file_0_view_0), for each $i$
- $w_i$: "weight" column in value.csv (table_id: file_0_view_0), for each $i$
- $W$: 15 (from user description)
- $x_i$: binary decision variable for each $i \in I$