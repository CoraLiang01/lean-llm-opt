## Mathematical Model

**Sets**  
Let $I$ be the set of items, with $I = \{1, 2, \ldots, 140\}$ (from the "item" column in value.csv).

**Parameters**  
For each $i \in I$:
- $v_i$: value of item $i$ (from "value" column, table_id: file_0_view_0)
- $w_i$: weight of item $i$ (from "weight" column, table_id: file_0_view_0)

Let $W = 15$ (total weight capacity).

**Decision Variables**  
For each $i \in I$:
- $x_i \in \{0,1\}$: $x_i = 1$ if item $i$ is selected, $0$ otherwise.

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

- $I$: All "item" values in table_id: file_0_view_0, column "item"
- $v_i$: table_id: file_0_view_0, column "value", row with item $i$
- $w_i$: table_id: file_0_view_0, column "weight", row with item $i$
- $W = 15$: from user description (merchandise counter holds 15 units of weight)
- $x_i$: binary variable for each $i \in I$

**Source Table:**  
- value.csv (table_id: file_0_view_0), columns: "item", "value", "weight"