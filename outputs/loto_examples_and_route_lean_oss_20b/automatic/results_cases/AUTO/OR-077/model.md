**Mathematical Model**

\[
\begin{aligned}
\text{Sets} \quad & I \;\; \text{(items)} \\[4pt]
\text{Parameters} \quad & \text{value}_i \;\; (i\in I) \\
& \text{weight}_i \;\; (i\in I) \\
& W \;\; = 15 \\[4pt]
\text{Decision Variables} \quad & x_i \in \{0,1\} \;\; (i\in I) \\[4pt]
\text{Objective} \quad & \max \sum_{i\in I} \text{value}_i\,x_i \\[4pt]
\text{Constraint} \quad & \sum_{i\in I} \text{weight}_i\,x_i \;\le\; W
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column |
|-----------|--------------|------------|--------------|
| `I` | `file_0_view_0` | `item` | – |
| `value_i` | `file_0_view_0` | `item` | `value` |
| `weight_i` | `file_0_view_0` | `item` | `weight` |
| `W` | – | – | 15 |

All indices and parameters are taken directly from the supplied `value.csv` data.