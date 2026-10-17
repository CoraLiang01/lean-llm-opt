**Mathematical Model**

Let  

- \(P\) be the set of all products (identified by the *Product Name* column).  
- For each \(p\in P\)  
  - \(r_p\) : revenue per unit (from the *Revenue* column).  
  - \(d_p\) : deterministic demand (from the *Demand* column).  
  - \(I_p\) : initial inventory (from the *Initial Inventory* column).  
- Decision variable \(x_p\) : number of units of product \(p\) to sell.

\[
\begin{aligned}
\max_{x}\;& \sum_{p\in P} r_p\,x_p \\[4pt]
\text{s.t. } & 0 \le x_p \le d_p && \forall p\in P \\[2pt]
             & 0 \le x_p \le I_p && \forall p\in P \\[2pt]
             & x_p \in \mathbb{Z}_{+} && \forall p\in P
\end{aligned}
\]

---

**Data Mapping**

| Table ID | Column in CSV | Parameter / Index |
|----------|---------------|-------------------|
| `file_0_view_0` | `Product Name` | index \(p\) |
| `file_0_view_0` | `Revenue` | \(r_p\) |
| `file_0_view_0` | `Demand` | \(d_p\) |
| `file_0_view_0` | `Initial Inventory` | \(I_p\) |

All parameters are taken directly from the corresponding columns of the CSV file; no numerical values are enumerated in the model.