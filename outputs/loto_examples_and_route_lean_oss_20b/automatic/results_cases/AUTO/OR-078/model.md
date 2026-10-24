**Mathematical Model**

Let  

* \(I\) – set of all generation lot options (identified by the column **option** in *energy.csv*).  
* \(g_i\) – generation produced by one lot of option \(i\) (column **gen_per_lot**).  
* \(c_i\) – cost of one lot of option \(i\) (column **cost_per_lot**).  
* \(x_i\) – number of lots of option \(i\) to purchase (decision variable).  

The required total demand is \(D = 200\).

\[
\begin{aligned}
\min \quad & \sum_{i\in I} c_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} g_i\,x_i \;\ge\; D \\[4pt]
& x_i \in \mathbb{Z}_{+}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(I\) | Index set | `file_0_view_0` | `option` |
| \(g_i\) | Generation per lot | `file_0_view_0` | `gen_per_lot` |
| \(c_i\) | Cost per lot | `file_0_view_0` | `cost_per_lot` |
| \(D\) | Demand | *constant* | 200 |

All parameters are taken directly from the complete runtime records of *energy.csv*; no additional data are introduced.