**Mathematical Model**

Let  

- \(I\) be the set of all items (indices taken from the column **item** of the table).  
- \(v_i\) be the value of item \(i\) (column **value**).  
- \(w_i\) be the weight of item \(i\) (column **weight**).  
- \(W = 15\) be the total weight capacity of the merchandise counter.  

Decision variable  

\[
x_i \in \{0,1\}\quad \forall i\in I
\]

---

**Objective**

\[
\max \;\; \sum_{i\in I} v_i\,x_i
\]

---

**Constraints**

1. **Weight capacity**

\[
\sum_{i\in I} w_i\,x_i \;\le\; W
\]

2. **Binary decision**

\[
x_i \in \{0,1\}\quad \forall i\in I
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(I\) | Item identifiers | `file_0_view_0` | `item` |
| \(v_i\) | Value of item | `file_0_view_0` | `value` |
| \(w_i\) | Weight of item | `file_0_view_0` | `weight` |
| \(W\) | Capacity constant | — | 15 |

All parameters are taken directly from the specified columns of the table `file_0_view_0`; no additional data or transformations are introduced.