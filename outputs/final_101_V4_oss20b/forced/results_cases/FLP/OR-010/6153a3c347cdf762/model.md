**Mathematical Model**

Let  

- \(I\) be the set of products (indexed by the *Product Name* values).  
- Parameters  
  - \(r_i\) – revenue per unit of product \(i\) (column **Revenue**).  
  - \(d_i\) – demand for product \(i\) (column **Demand**).  
  - \(s_i\) – initial inventory of product \(i\) (column **Initial Inventory**).  
- Decision variable  
  - \(x_i \in \mathbb{Z}_{\ge 0}\) – quantity of product \(i\) that is fulfilled.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & x_i \;\le\; d_i &&\forall i\in I \quad\text{(demand limit)}\\
            & x_i \;\le\; s_i &&\forall i\in I \quad\text{(inventory limit)}\\
            & x_i \in \mathbb{Z}_{\ge 0} &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(r_i\)   | `file_0_view_0` | `Revenue` |
| \(d_i\)   | `file_0_view_0` | `Demand` |
| \(s_i\)   | `file_0_view_0` | `Initial Inventory` |

The index set \(I\) is the set of all distinct values in the **Product Name** column of `file_0_view_0`. No other data are required.