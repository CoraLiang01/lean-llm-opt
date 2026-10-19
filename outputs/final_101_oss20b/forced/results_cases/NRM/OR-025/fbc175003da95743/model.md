**Abstract Optimization Model**

**Index Set**  
\(I = \{\, i \mid \text{Product Name of record } i \text{ contains the string } \text{TABLET}\,\}\)

**Parameters**  
\[
\begin{aligned}
w_i &\;=\; \text{Revenue of product } i, \qquad &\forall i \in I\\
d_i &\;=\; \text{Demand of product } i, \qquad &\forall i \in I\\
c_i &\;=\; \text{Initial Inventory of product } i, \qquad &\forall i \in I
\end{aligned}
\]

**Decision Variables**  
\[
x_i \;\in\; \mathbb{Z}, \qquad x_i \;\ge\; 0, \qquad \forall i \in I
\]

**Objective Function**  
\[
\max \;\sum_{i \in I} w_i \, x_i
\]

**Constraints**  

1. **Inventory Constraints**  
\[
x_i \;\le\; c_i, \qquad \forall i \in I
\]

2. **Demand Constraints**  
\[
x_i \;\le\; d_i, \qquad \forall i \in I
\]

---

**Data Mapping**

| Parameter | Source Table ID | Source Column |
|-----------|-----------------|---------------|
| \(w_i\)   | `file_0_view_0` | `Revenue`     |
| \(d_i\)   | `file_0_view_0` | `Demand`      |
| \(c_i\)   | `file_0_view_0` | `Initial Inventory` |

*All records in `file_0_view_0` whose `Product Name` contains the substring “TABLET” constitute the index set \(I\).*