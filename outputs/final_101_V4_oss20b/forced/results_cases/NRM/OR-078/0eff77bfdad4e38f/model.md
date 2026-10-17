**Mathematical Model**

Let  

- \(I\) – set of all generation lot options (identified by the column *option*).  
- For each \(i\in I\)  
  - \(g_i\) – generation per lot (column *gen_per_lot*).  
  - \(c_i\) – cost per lot (column *cost_per_lot*).  

Decision variable  

- \(x_i \in \mathbb{Z}_{\ge 0}\) – number of lots of option \(i\) to purchase.

\[
\begin{aligned}
\min \quad & \sum_{i\in I} c_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} g_i\,x_i \;\ge\; 200 \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}, \qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(g_i\) | Generation per lot | `file_0_view_0` | `gen_per_lot` |
| \(c_i\) | Cost per lot | `file_0_view_0` | `cost_per_lot` |
| \(I\) | Option identifiers | `file_0_view_0` | `option` |

All values for the parameters are taken directly from the complete runtime records of `energy.csv` as presented in the observation. No additional data or constraints are introduced beyond those specified in the query.