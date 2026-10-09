## Mathematical Model

**Sets**  
Let $I$ be the set of items, with $I = \{\text{all item values in column "item" of file_0_view_0}\}$.

**Parameters**  
For each $i \in I$:
- $v_i$: value of item $i$ (from column "value" in file_0_view_0)
- $w_i$: weight of item $i$ (from column "weight" in file_0_view_0)

Let $W = 15$ (total weight capacity)

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

$$
x_i \in \{0,1\} \quad \forall i \in I
$$

---

### Data Mapping

- $I$: All values in column "item" of table_id file_0_view_0 (value.csv)
- $v_i$: column "value" in file_0_view_0, for each $i$
- $w_i$: column "weight" in file_0_view_0, for each $i$
- $W = 15$ (from user description)

**Source Table:**  
- file_0_view_0: value.csv, columns ["item", "value", "weight"]