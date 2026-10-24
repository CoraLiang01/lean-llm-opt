**Mathematical Model**

Let  

- \(I\) be the set of products returned by the query (two 27in products).  
- For each \(i\in I\)  
  - \(R_i\) – revenue per unit (from column **Revenue**)  
  - \(D_i\) – total demand over the horizon (from column **Demand**)  
  - \(S_i\) – initial inventory (from column **Initial Inventory**)  
  - \(x_i\) – number of units to sell (decision variable)

\[
\begin{aligned}
\max\;& \sum_{i\in I} R_i\,x_i \\[4pt]
\text{s.t. } & x_i \;\le\; D_i && \forall i\in I \\[2pt]
             & x_i \;\le\; S_i && \forall i\in I \\[2pt]
             & x_i \in \mathbb{Z}_{+} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(R_i\)   | `file_0_view_0` | `Revenue` |
| \(D_i\)   | `file_0_view_0` | `Demand` |
| \(S_i\)   | `file_0_view_0` | `Initial Inventory` |
| \(x_i\)   | – (decision variable) | – |

*Index set \(I\) is derived from all rows returned by the filter on “27in” in `file_0_view_0`.*