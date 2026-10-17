#### Index Sets
- $I$: set of items (from column "item" in table_id: file_0_view_0)

#### Parameters
- $v_i$: value of item $i$ (from column "value" in table_id: file_0_view_0)
- $w_i$: weight of item $i$ (from column "weight" in table_id: file_0_view_0)
- $W$: total weight capacity of the merchandise counter ($W = 15$)

#### Decision Variables
- $x_i \in \{0,1\}$: 1 if item $i$ is selected for display, 0 otherwise, for all $i \in I$

#### Objective
$$
\max \sum_{i \in I} v_i x_i
$$

#### Constraints
1. Weight capacity:
$$
\sum_{i \in I} w_i x_i \leq W
$$

2. Binary selection:
$$
x_i \in \{0,1\} \quad \forall i \in I
$$

---

#### Data Mapping

- Table: value.csv (table_id: file_0_view_0)
    - Index set $I$: all rows, column "item"
    - Parameter $v_i$: column "value"
    - Parameter $w_i$: column "weight"
- Weight capacity $W$ is given in the user query (not in the table).