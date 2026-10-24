**Mathematical Model**

Let  

* \(P\) – set of products (indexed by product name).  
* \(R_p\) – revenue per unit of product \(p\).  
* \(D_p\) – deterministic demand of product \(p\).  
* \(I_p\) – initial inventory of product \(p\).  
* \(x_p\) – number of units of product \(p\) to be sold (decision variable).

\[
\begin{aligned}
\max\;& \sum_{p\in P} R_p\,x_p \\[4pt]
\text{s.t. } & 0 \le x_p \le D_p &&\forall p\in P \\[2pt]
             & 0 \le x_p \le I_p &&\forall p\in P \\[2pt]
             & x_p \ge 0 &&\forall p\in P
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(P\) | Product identifiers | `file_0_view_0` | `Product Name` |
| \(R_p\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(D_p\) | Demand quantity | `file_0_view_0` | `Demand` |
| \(I_p\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |

Each record in `file_0_view_0` supplies a single product \(p\) with the corresponding values for \(R_p\), \(D_p\), and \(I_p\). The decision variable \(x_p\) is defined for every product \(p\) in the index set \(P\).