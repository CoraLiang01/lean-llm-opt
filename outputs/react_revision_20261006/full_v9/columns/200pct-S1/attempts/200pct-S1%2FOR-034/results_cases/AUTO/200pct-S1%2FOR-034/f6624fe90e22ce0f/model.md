## Mathematical Model

**Sets**  
Let $I$ be the set of items, indexed by $i$.  
From Data Mapping: $I = \{\text{all item values in column "item" of table_id file_0_view_0}\}$

**Parameters**  
For each $i \in I$:
- $v_i$: value of item $i$ (from column "value", table_id file_0_view_0)
- $w_i$: weight of item $i$ (from column "weight", table_id file_0_view_0)
- $W = 15$: total weight capacity

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

- $I$: All rows in column "item" of table_id file_0_view_0 (value.csv)
- $v_i$: column "value" in table_id file_0_view_0, matched by $i$
- $w_i$: column "weight" in table_id file_0_view_0, matched by $i$
- $W$: scalar, 15 (from user description)
- $x_i$: binary variable for each $i \in I$