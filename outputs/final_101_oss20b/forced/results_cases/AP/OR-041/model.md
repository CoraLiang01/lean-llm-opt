**Mathematical Model**

Let  

- \(I\) be the set of real‑estate areas (rows of *products.csv*).  
- For each \(i\in I\)  
  - \(v_i\) : benefit per unit of development (column **Value**).  
  - \(w_i\) : resource consumption per unit of development (column **Weight**).  
- \(C\) : total development capacity (column **Capacity** of *capacity.csv*).  
- \(x_i\) : scale of development per day in area \(i\) (decision variable).

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\ge\; 0 \qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(v_i\)   | `file_1_view_0` | `Value` |
| \(w_i\)   | `file_1_view_0` | `Weight` |
| \(C\)     | `file_0_view_0` | `Capacity` |

*Index set* \(I\) is derived from all rows returned in `file_1_view_0`.