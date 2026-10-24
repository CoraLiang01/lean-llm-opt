**Mathematical Model**

Let  

- \(I\) be the set of real‑estate areas (ProductName).  
- \(v_i\) be the benefit per unit of development in area \(i\).  
- \(w_i\) be the development‑capacity weight per unit in area \(i\).  
- \(C\) be the overall development‑capacity limit.  
- \(x_i \ge 0\) be the scale of development per day in area \(i\).

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\ge\; 0 \qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(I\) | `file_1_view_0` | `ProductName` |
| \(v_i\) | `file_1_view_0` | `Value` |
| \(w_i\) | `file_1_view_0` | `Weight` |
| \(C\) | `file_0_view_0` | `Capacity` |

All indices and parameters are taken directly from the corresponding CSV files; no additional numerical values are enumerated.