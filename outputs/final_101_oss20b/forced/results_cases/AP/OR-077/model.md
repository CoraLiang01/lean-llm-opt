**Mathematical Model**

Let  

- \(I\) be the set of items.  
- \(x_i \in \{0,1\}\) be the decision variable indicating whether item \(i\) is selected.  
- \(v_i\) be the value of item \(i\).  
- \(w_i\) be the weight of item \(i\).  
- \(W = 15\) be the weight capacity of the merchandise counter.

\[
\begin{aligned}
\max \quad & \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} w_i\,x_i \;\le\; W \\[4pt]
& x_i \in \{0,1\} \qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(v_i\) | Value | `file_0_view_0` | `value` |
| \(w_i\) | Weight | `file_0_view_0` | `weight` |
| \(I\) | Items | `file_0_view_0` | `item` |

*All indices \(i\) are taken from the `item` column of the table `file_0_view_0`.  No numerical values are enumerated in the model; they are retrieved from the specified columns.*