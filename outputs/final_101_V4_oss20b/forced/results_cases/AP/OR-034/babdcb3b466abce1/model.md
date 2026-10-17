**Mathematical Model**

Let  

* \(I\) – set of products (indexed by the values in the *Product Name* column).  
* \(r_i\) – revenue per unit of product \(i\) (from the *Revenue* column).  
* \(d_i\) – demand for product \(i\) (from the *Demand* column).  
* \(s_i\) – initial inventory of product \(i\) (from the *Initial Inventory* column).  

Decision variable  

* \(x_i\) – quantity of product \(i\) to sell (integer, \(x_i\ge 0\)).  

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& x_i \;\le\; d_i && \forall i\in I \\[2pt]
            & x_i \;\le\; s_i && \forall i\in I \\[2pt]
            & x_i \;\in\; \mathbb{Z}_{+} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(r_i\)   | `file_0_view_0` | `Revenue` |
| \(d_i\)   | `file_0_view_0` | `Demand` |
| \(s_i\)   | `file_0_view_0` | `Initial Inventory` |
| Index set \(I\) | `file_0_view_0` | `Product Name` |

All values are taken directly from the CSV file `Frenchbakerydailysales.csv`; no numerical values are enumerated in the model.