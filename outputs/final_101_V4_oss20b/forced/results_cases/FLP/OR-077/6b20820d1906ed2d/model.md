**Mathematical Model**

Let  

- \(I\) be the set of items.  
- \(x_i \in \{0,1\}\) be the binary decision variable indicating whether item \(i\) is selected.  
- \(v_i\) be the value of item \(i\).  
- \(w_i\) be the weight of item \(i\).  
- \(W = 15\) be the weight capacity of the merchandise counter.

\[
\begin{aligned}
\max \quad & \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} w_i\,x_i \;\le\; W \\[4pt]
& x_i \in \{0,1\} \quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(v_i\) | value | `file_0_view_0` | `value` |
| \(w_i\) | weight | `file_0_view_0` | `weight` |
| \(I\) | items | `file_0_view_0` | `item` |

The capacity \(W\) is a constant given in the problem statement. No other data are required.