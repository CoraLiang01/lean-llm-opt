**Mathematical Model**

Let  

* \(I\) – set of Fashion products (identified by the value in column **Product Name**).  
* For each \(i\in I\)  
  * \(p_i\) – revenue per unit (column **Revenue**).  
  * \(d_i\) – maximum demand (column **Demand**).  
  * \(s_i\) – initial inventory (column **Initial Inventory**).  

Decision variable  

* \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) to sell.

\[
\begin{aligned}
\max\;& \sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t.}\;& x_i \;\le\; d_i && \forall i\in I \\[2pt]
& x_i \;\le\; s_i && \forall i\in I \\[2pt]
& x_i \;\ge\; 0 && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | `file_0_view_0` | `Product Name` |
| \(p_i\) | `file_0_view_0` | `Revenue` |
| \(d_i\) | `file_0_view_0` | `Demand` |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` |
| \(x_i\) | – | – |

All parameters are taken directly from the corresponding columns of the filtered view `file_0_view_0`. No additional data or assumptions are introduced.