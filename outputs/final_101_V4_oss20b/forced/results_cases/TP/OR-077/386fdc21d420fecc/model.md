**Mathematical Model**

Let  

- \(I=\{1,2,\dots ,140\}\) (index of items)  
- \(x_i\in\{0,1\}\) (1 if item \(i\) is selected, 0 otherwise)  

Parameters (obtained from *value.csv*)  

- \(v_i\) value of item \(i\)  
- \(w_i\) weight of item \(i\)  

Capacity  

- \(W=15\) (total weight allowed)

\[
\begin{aligned}
\max\;&\sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t. } &\sum_{i\in I} w_i\,x_i \;\le\; W \\[4pt]
&x_i \in \{0,1\}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(v_i\)   | value.csv | file_0_view_0 | `value` |
| \(w_i\)   | value.csv | file_0_view_0 | `weight` |
| \(i\)     | value.csv | file_0_view_0 | `item` (converted to integer index) |

The index set \(I\) is the set of all distinct `item` values in *value.csv* (1–140).  
The capacity \(W\) is fixed at 15 as stated in the problem description.