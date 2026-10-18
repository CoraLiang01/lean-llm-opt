##### Abstract Model

**Index Set**  
\(I\) – set of items indexed by the column *item* in *value.csv*.

**Parameters**  
\(\text{value}_i\) – value of item \(i\) (from column *value*).  
\(\text{weight}_i\) – weight of item \(i\) (from column *weight*).  
\(W = 15\) – total weight capacity of the merchandise counter.

**Decision Variables**  
\(y_i \in \{0,1\}\) – binary variable indicating whether item \(i\) is selected.

**Objective Function**  
\[
\quad \max \quad \sum_{i \in I} \text{value}_i \cdot y_i
\]

**Constraints**  

1. **Weight Capacity**  
\[
\sum_{i \in I} \text{weight}_i \cdot y_i \;\le\; W
\]

2. **Binary Decision Variables**  
\[
y_i \in \{0,1\}, \quad \forall i \in I
\]

---

#### Data Mapping

| Parameter | Source Table | Source Column |
|-----------|---------------|---------------|
| \(\text{value}_i\) | `file_0_view_0` | `value` |
| \(\text{weight}_i\) | `file_0_view_0` | `weight` |
| \(W\) | constant | 15 |

*The table `file_0_view_0` corresponds to `value.csv` and contains the columns `item`, `value`, and `weight` for all 140 items.*