## Mathematical Model

**Sets**  
Let $I$ be the set of items, with $I = \{1, 2, \ldots, 140\}$ (from column "item" in value.csv).

**Parameters**  
For each $i \in I$:
- $v_i$: value of item $i$ (column "value", table_id: file_0_view_0)
- $w_i$: weight of item $i$ (column "weight", table_id: file_0_view_0)

Let $W = 15$ (total weight capacity, from the question).

**Decision Variables**  
For each $i \in I$:
- $x_i \in \{0,1\}$: 1 if item $i$ is selected, 0 otherwise

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

- $I$: All rows in value.csv, column "item", table_id: file_0_view_0
- $v_i$: value.csv, column "value", table_id: file_0_view_0, for each $i$
- $w_i$: value.csv, column "weight", table_id: file_0_view_0, for each $i$
- $W$: 15 (from question)

**All 140 items in value.csv are included.**