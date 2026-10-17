**Mathematical Model**

Let  

* \(I\) – set of all products (identified by *Product Name*).  
* For each \(p\in I\)  
  * \(r_p\) – revenue per unit (column **Revenue**).  
  * \(d_p\) – demand quantity (column **Demand**).  
  * \(i_p\) – initial inventory (column **Initial Inventory**).  

Decision variable  

* \(x_p\ge 0\) – quantity of product \(p\) that the bakery plans to sell.

\[
\begin{aligned}
\max\;&\sum_{p\in I} r_p\,x_p \\[4pt]
\text{s.t.}\;&x_p \;\le\; d_p &&\forall p\in I \\[2pt]
&x_p \;\le\; i_p &&\forall p\in I \\[2pt]
&x_p \;\ge\; 0 &&\forall p\in I
\end{aligned}
\]

---

**Data‑Mapping Rules**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(I\) | Set of product identifiers | `file_0_view_0` | `Product Name` |
| \(r_p\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_p\) | Demand quantity | `file_0_view_0` | `Demand` |
| \(i_p\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |

All parameters are taken directly from the corresponding columns of the CSV file `Frenchbakerydailysales.csv`. No numerical values are enumerated in the model; the mapping rules provide the exact source of each coefficient.