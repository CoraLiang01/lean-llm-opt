**Mathematical Model**

Let  

- \(I\) – set of vehicle types (indexed by the rows of *products.csv*).  

Parameters  

- \(p_i\) – profit per unit of type \(i\) (column **Value**).  
- \(w_i\) – weight (inventory space) per unit of type \(i\) (column **Weight**).  
- \(C\) – total inventory capacity (column **Capacity**).  

Decision variables  

- \(x_i \in \mathbb{Z}_{\ge 0}\) – number of vehicles of type \(i\) to order per day.  

\[
\begin{aligned}
\max \quad & \sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\ge\; 0,\; x_i \in \mathbb{Z}\quad \forall i\in I .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(p_i\) | `file_1_view_0` | `Value` |
| \(w_i\) | `file_1_view_0` | `Weight` |
| \(C\)   | `file_0_view_0` | `Capacity` |

All parameters are taken directly from the specified CSV files; no additional data or assumptions are introduced.