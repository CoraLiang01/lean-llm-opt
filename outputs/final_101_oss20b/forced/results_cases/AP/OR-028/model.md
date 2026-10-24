**Mathematical Model**

Let  

- \(I\) be the set of products (indexed by the *Product Name* values).  
- For each \(i\in I\)  
  - \(r_i\) : revenue per unit (from column **Revenue**)  
  - \(d_i\) : demand quantity (from column **Demand**)  
  - \(s_i\) : initial inventory (from column **Initial Inventory**)  

Decision variable  

- \(x_i \in \mathbb{Z}_{\ge 0}\) : number of units of product \(i\) to fulfill.

\[
\begin{aligned}
\max \quad & \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\quad & x_i \;\le\; d_i \qquad &&\forall i\in I \\[2pt]
& x_i \;\le\; s_i \qquad &&\forall i\in I \\[2pt]
& x_i \;\ge\; 0 \qquad &&\forall i\in I \\[2pt]
& x_i \in \mathbb{Z} \qquad &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(r_i\)   | file_0_view_0 | Revenue |
| \(d_i\)   | file_0_view_0 | Demand |
| \(s_i\)   | file_0_view_0 | Initial Inventory |
| \(I\)     | file_0_view_0 | Product Name |

All indices and parameters are taken directly from the specified CSV file; no numerical values are enumerated in the model.