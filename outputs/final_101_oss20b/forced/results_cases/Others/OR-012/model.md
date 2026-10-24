**Mathematical Model**

Let  

* \(I\) be the set of all products in *OnlineSalesDataset.csv* (119 items).  
* For each \(p\in I\)  
  * \(r_p\) – revenue per unit (RMB)  
  * \(d_p\) – deterministic demand (units)  
  * \(i_p\) – initial inventory (units)  

Decision variable  

* \(x_p\) – number of units of product \(p\) to sell (integer, \(x_p\ge 0\))  

\[
\begin{aligned}
\max_{x}\;&\sum_{p\in I} r_p\,x_p \\[4pt]
\text{s.t.}\;&0 \le x_p \le d_p &&\forall p\in I \\[2pt]
&0 \le x_p \le i_p &&\forall p\in I \\[2pt]
&x_p \in \mathbb{Z}_{\ge 0} &&\forall p\in I
\end{aligned}
\]

The two upper‑bound constraints together enforce  
\(x_p \le \min\{d_p,i_p\}\).

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(r_p\) | `file_0_view_0` | `Revenue` |
| \(d_p\) | `file_0_view_0` | `Demand` |
| \(i_p\) | `file_0_view_0` | `Initial Inventory` |

| Variable | Domain |
|----------|--------|
| \(x_p\) | integer, \(x_p\ge 0\) |

The index set \(I\) is defined as the set of all distinct *Product Name* values returned in the observation. No additional filtering is applied.