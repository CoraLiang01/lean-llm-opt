**Mathematical Model**

Let  

- \(P\) be the set of product names (pizza types) extracted from the table.  

Parameters (indexed by \(p\in P\))  

- \(r_p\) – revenue per unit of product \(p\) (from column **Revenue**).  
- \(d_p\) – demand for product \(p\) (from column **Demand**).  
- \(i_p\) – initial inventory of product \(p\) (from column **Initial Inventory**).  

Decision variable  

- \(x_p \in \mathbb{Z}_{+}\) – number of units of product \(p\) to sell.  

**Objective**  

\[
\max \sum_{p\in P} r_p\,x_p
\]

**Constraints**  

\[
\begin{aligned}
&x_p \le d_p &&\forall p\in P &&\text{(demand limit)}\\
&x_p \le i_p &&\forall p\in P &&\text{(inventory limit)}\\
&x_p \in \mathbb{Z}_{+} &&\forall p\in P
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(r_p\)   | file_0_view_0 | Revenue |
| \(d_p\)   | file_0_view_0 | Demand |
| \(i_p\)   | file_0_view_0 | Initial Inventory |

The index set \(P\) is the set of distinct values in the **Product Name** column of `file_0_view_0`. No other data are required.