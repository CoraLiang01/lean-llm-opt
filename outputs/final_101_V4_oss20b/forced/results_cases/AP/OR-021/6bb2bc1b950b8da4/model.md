**Mathematical Model**

Let  

- \(P\) be the set of all products (identified by the *Product Name* column).  
- For each \(p\in P\)  
  - \(r_p\)  – revenue per unit (from *Revenue*).  
  - \(d_p\)  – deterministic demand (from *Demand*).  
  - \(i_p\)  – initial inventory (from *Initial Inventory*).  
- Decision variable  
  - \(x_p\in \mathbb{Z}_{+}\) – number of units of product \(p\) to fulfill.

\[
\begin{aligned}
\max\;& \sum_{p\in P} r_p\,x_p \\[4pt]
\text{s.t. } & 0 \le x_p \le d_p && \forall p\in P \\[2pt]
             & 0 \le x_p \le i_p && \forall p\in P \\[2pt]
             & x_p \in \mathbb{Z}_{+} && \forall p\in P
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Column | Table ID |
|-----------|----------------|----------|
| \(r_p\)   | *Revenue*      | file_0_view_0 |
| \(d_p\)   | *Demand*       | file_0_view_0 |
| \(i_p\)   | *Initial Inventory* | file_0_view_0 |
| Index set \(P\) | *Product Name* | file_0_view_0 |

All values are taken directly from the CSV rows; no enumeration of individual rows is shown.