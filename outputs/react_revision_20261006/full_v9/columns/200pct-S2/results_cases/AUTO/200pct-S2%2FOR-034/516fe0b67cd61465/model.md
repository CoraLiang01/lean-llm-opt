## Mathematical Model

**Sets**  
Let $I$ be the set of items, with $I = \{\text{1}, \text{2}, \ldots, \text{140}\}$ (from column "item" in value.csv).

**Parameters**  
For each $i \in I$:
- $v_i$: value of item $i$ (from column "value", table_id: file_0_view_0)
- $w_i$: weight of item $i$ (from column "weight", table_id: file_0_view_0)

Let $W = 15$ (total weight capacity).

**Decision Variables**  
For each $i \in I$:
- $x_i \in \{0,1\}$: $1$ if item $i$ is selected, $0$ otherwise.

**Objective**  
Maximize total value:
$$
\max \sum_{i \in I} v_i x_i
$$

**Constraints**  
Total weight does not exceed capacity:
$$
\sum_{i \in I} w_i x_i \leq W
$$

$$
x_i \in \{0,1\} \quad \forall i \in I
$$

---

### Data Mapping

- $I$: All "item" values in value.csv (table_id: file_0_view_0, column "item")
- $v_i$: value.csv, column "value", table_id: file_0_view_0, for each $i$
- $w_i$: value.csv, column "weight", table_id: file_0_view_0, for each $i$
- $W = 15$: merchandise counter weight limit (from user description)

**All parameters and sets are mapped directly from the current value.csv as described above.**