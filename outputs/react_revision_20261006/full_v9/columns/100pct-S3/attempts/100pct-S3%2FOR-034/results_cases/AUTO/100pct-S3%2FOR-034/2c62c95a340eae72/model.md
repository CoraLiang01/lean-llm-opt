## Mathematical Model

**Sets**  
Let $I$ be the set of items, with $I = \{\text{1}, \text{2}, \ldots, \text{140}\}$ (from column "item" in value.csv).

**Parameters**  
For each $i \in I$:
- $v_i$: value of item $i$ (from column "value", table_id: file_0_view_0)
- $w_i$: weight of item $i$ (from column "weight", table_id: file_0_view_0)
- $W = 15$: total weight capacity

**Decision Variables**  
For each $i \in I$:
- $x_i \in \{0,1\}$: $x_i = 1$ if item $i$ is selected, $0$ otherwise

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

Binary selection:
$$
x_i \in \{0,1\} \quad \forall i \in I
$$

---

### Data Mapping

- $I$: All "item" values in value.csv (table_id: file_0_view_0, column "item")
- $v_i$: "value" column in value.csv (table_id: file_0_view_0, column "value")
- $w_i$: "weight" column in value.csv (table_id: file_0_view_0, column "weight")
- $W$: 15 (from user description)
- $x_i$: binary variable for each $i \in I$