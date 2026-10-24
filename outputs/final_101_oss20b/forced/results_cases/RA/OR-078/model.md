**Mathematical Model**

Let  

* \(I\) be the set of all generation options (rows of *energy.csv*).  
* For each \(i\in I\)  
  * \(g_i\) – generation per lot (units) – parameter from column **gen_per_lot**.  
  * \(c_i\) – cost per lot – parameter from column **cost_per_lot**.  
* Decision variable  
  * \(x_i \in \mathbb{Z}_{\ge 0}\) – number of lots of option \(i\) to purchase.

\[
\begin{aligned}
\min\;& \sum_{i\in I} c_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} g_i\,x_i \;\ge\; 200 \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(g_i\)   | file_0_view_0 | gen_per_lot |
| \(c_i\)   | file_0_view_0 | cost_per_lot |
| \(x_i\)   | – | – |

*Index set \(I\) is derived from the distinct values in the **option** column of *energy.csv* (all 131 rows).*