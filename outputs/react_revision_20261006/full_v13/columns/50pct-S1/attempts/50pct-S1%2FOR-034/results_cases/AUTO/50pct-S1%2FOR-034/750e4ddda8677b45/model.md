## Symbolic Mathematical Model

**Sets**  
$I$ : set of items, indexed by $i$ (from value.csv, column "item")

**Parameters**  
$v_i$ : value of item $i$ (from value.csv, column "value", table_id: file_0_view_0)  
$w_i$ : weight of item $i$ (from value.csv, column "weight", table_id: file_0_view_0)  
$W = 15$ : total weight capacity

**Decision Variables**  
$x_i \in \{0,1\}$ : 1 if item $i$ is selected, 0 otherwise

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

## Data Mapping

- $I$ = all items in value.csv, table_id: file_0_view_0, column "item"
- $v_i$ = value.csv, table_id: file_0_view_0, column "value", row $i$
- $w_i$ = value.csv, table_id: file_0_view_0, column "weight", row $i$
- $W = 15$ (from user description)
- $x_i$ : binary variable for each $i \in I$