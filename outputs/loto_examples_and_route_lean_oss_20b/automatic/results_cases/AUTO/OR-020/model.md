**Mathematical Model**

Let  

* \(I\) be the set of products indexed by the values in the column **Product Name** of table **file_0_view_0**.  

* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (column **Revenue**)  
  * \(d_i\) – demand (column **Demand**)  
  * \(s_i\) – initial inventory (column **Initial Inventory**)  

Decision variable  

* \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) to fulfill.

---

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& x_i \;\le\; d_i &&\forall i\in I \\[2pt]
& x_i \;\le\; s_i &&\forall i\in I \\[2pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0} &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column Name |
|--------|--------------|-------------|
| \(I\) | file_0_view_0 | **Product Name** |
| \(r_i\) | file_0_view_0 | **Revenue** |
| \(d_i\) | file_0_view_0 | **Demand** |
| \(s_i\) | file_0_view_0 | **Initial Inventory** |

All parameters are taken directly from the corresponding columns of the single table **file_0_view_0**. No additional data or assumptions are introduced.