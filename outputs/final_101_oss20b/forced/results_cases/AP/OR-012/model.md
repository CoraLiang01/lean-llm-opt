##### Model

Let  

- \(P\) be the set of products (identified by the *Product Name* column).  
- For each \(p\in P\)  
  - \(r_p\)  – revenue per unit (from *Revenue*).  
  - \(d_p\)  – known demand (from *Demand*).  
  - \(I_p\)  – initial inventory (from *Initial Inventory*).  

Decision variable  

- \(x_p\in \mathbb{Z}_{\ge 0}\) – units of product \(p\) to sell.

**Objective**

\[
\max \;\sum_{p\in P} r_p\,x_p
\]

**Constraints**

\[
\begin{aligned}
&x_p \;\le\; d_p &&\forall p\in P &&\text{(cannot sell more than demand)}\\
&x_p \;\le\; I_p &&\forall p\in P &&\text{(cannot sell more than inventory)}\\
&x_p \;\ge\; 0 &&\forall p\in P &&\text{(non‑negativity)}\\
&x_p \in \mathbb{Z} &&\forall p\in P &&\text{(integer units)}
\end{aligned}
\]

---

##### Data Mapping

| Table ID | Column in CSV | Parameter / Variable |
|----------|---------------|----------------------|
| file_0_view_0 | *Product Name* | index set \(P\) |
| file_0_view_0 | *Revenue* | \(r_p\) |
| file_0_view_0 | *Demand* | \(d_p\) |
| file_0_view_0 | *Initial Inventory* | \(I_p\) |

No numerical values are enumerated; each parameter is linked directly to the corresponding CSV column.