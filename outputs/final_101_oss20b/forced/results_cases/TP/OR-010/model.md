**Mathematical model**

Let  

* \(I\) – set of products (identified by *Product Name*).  
* \(r_i\) – revenue per unit of product \(i\) (column **Revenue**).  
* \(d_i\) – demand of product \(i\) (column **Demand**).  
* \(s_i\) – initial inventory of product \(i\) (column **Initial Inventory**).  
* \(x_i\) – quantity of product \(i\) that is fulfilled (decision variable).

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & x_i \;\le\; d_i &&\forall i\in I \quad\text{(do not exceed demand)}\\
            & x_i \;\le\; s_i &&\forall i\in I \quad\text{(do not exceed inventory)}\\
            & x_i \;\in\; \mathbb{Z}_{\ge 0} &&\forall i\in I \quad\text{(integer, non‑negative)}
\end{aligned}
\]

No restocking or in‑transit inventory is allowed, so the two upper‑bound constraints are sufficient.

---

**Data‑mapping rules**

| Parameter | Source column | Table ID | Mapping rule |
|-----------|---------------|----------|--------------|
| \(I\) | *Product Name* | `file_0_view_0` | Each distinct value of **Product Name** becomes an index \(i\). |
| \(r_i\) | **Revenue** | `file_0_view_0` | For each row, assign \(r_i =\) numeric value of **Revenue**. |
| \(d_i\) | **Demand** | `file_0_view_0` | For each row, assign \(d_i =\) numeric value of **Demand**. |
| \(s_i\) | **Initial Inventory** | `file_0_view_0` | For each row, assign \(s_i =\) numeric value of **Initial Inventory**. |

All numeric values are taken as real numbers; the decision variables \(x_i\) are declared integer and non‑negative. No additional data or constraints are required.