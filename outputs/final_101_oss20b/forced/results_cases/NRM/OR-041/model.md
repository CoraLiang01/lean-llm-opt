**Symbolic Model**

Let  

- \(I\) – set of real‑estate areas (rows of *products.csv*).  
- \(c\) – total development capacity (value in *capacity.csv*).  

Parameters (indexed by \(i\in I\))  

- \(v_i\) – development benefit per unit in area \(i\) (column **Value**).  
- \(w_i\) – development weight (resource consumption) per unit in area \(i\) (column **Weight**).  

Decision variables  

- \(x_i \in \mathbb{Z}_{\ge 0}\) – scale of development per day in area \(i\).  

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; c \\[4pt]
& x_i \;\ge\; 0,\; x_i \in \mathbb{Z}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(c\) | `file_0_view_0` | `Capacity` |
| \(v_i\) | `file_1_view_0` | `Value` |
| \(w_i\) | `file_1_view_0` | `Weight` |
| \(I\) | `file_1_view_0` | `ProductName` |

All values are taken directly from the corresponding CSV files; no additional data are introduced.