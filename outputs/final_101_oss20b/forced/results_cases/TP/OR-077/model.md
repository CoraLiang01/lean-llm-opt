**Mathematical model**

Let  

* \(I\) – set of items (indices taken from the rows of *value.csv*).  
* \(v_i\) – value of item \(i\).  
* \(w_i\) – weight of item \(i\).  
* \(x_i \in \{0,1\}\) – binary decision variable indicating whether item \(i\) is selected.

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} w_i\,x_i \;\le\; 15 \\[4pt]
& x_i \in \{0,1\}\quad \forall i\in I
\end{aligned}
\]

---

**Data‑mapping rules**

| Parameter | Source table | Column in source | Mapping rule |
|-----------|--------------|------------------|--------------|
| \(I\) | `file_0_view_0` | `item` | Each distinct value of the `item` column becomes an index in \(I\). |
| \(v_i\) | `file_0_view_0` | `value` | For each row, assign \(v_{\text{item}} = \text{value}\). |
| \(w_i\) | `file_0_view_0` | `weight` | For each row, assign \(w_{\text{item}} = \text{weight}\). |

All numerical values are taken verbatim from the corresponding columns of the CSV file; no aggregation or transformation is performed. The capacity limit \(15\) is a constant specified in the problem statement.