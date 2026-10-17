**Mathematical Model**

Let  

- \(I\) be the set of items (indices from the table).  
- \(v_i\) be the value of item \(i\).  
- \(w_i\) be the weight of item \(i\).  
- \(x_i \in \{0,1\}\) be the binary decision variable indicating whether item \(i\) is selected.

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; 15 \\[4pt]
& x_i \in \{0,1\}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(I\) | Index set | `file_0_view_0` | `item` |
| \(v_i\) | Value | `file_0_view_0` | `value` |
| \(w_i\) | Weight | `file_0_view_0` | `weight` |

The capacity constant \(15\) is given in the problem statement and is not derived from the table.