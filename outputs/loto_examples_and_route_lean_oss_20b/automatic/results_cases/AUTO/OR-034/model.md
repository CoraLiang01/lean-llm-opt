**Mathematical Model**

Let  

* \(I\) – set of all products (identified by *Product Name*).  
* \(r_i\) – revenue per unit of product \(i\) (column **Revenue**).  
* \(inv_i\) – initial inventory of product \(i\) (column **Initial Inventory**).  
* \(d_i\) – demand for product \(i\) (column **Demand**).  

Decision variable  

* \(x_i \in \mathbb{Z}_{\ge 0}\) – quantity of product \(i\) to sell.

---

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\
\text{s.t. } & 0 \le x_i \le inv_i && \forall i\in I \\
& 0 \le x_i \le d_i   && \forall i\in I \\
& x_i \in \mathbb{Z}  && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Index | Parameter | Source Table | Source Column |
|-------|-----------|--------------|---------------|
| \(I\) | – | `file_0_view_0` | `Product Name` |
| \(r_i\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(inv_i\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |
| \(d_i\) | Demand | `file_0_view_0` | `Demand` |

All parameters are taken directly from the CSV view `file_0_view_0`; no additional data are introduced.