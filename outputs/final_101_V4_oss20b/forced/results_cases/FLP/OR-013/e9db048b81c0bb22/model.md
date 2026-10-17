**Mathematical Model**

Let  

* \(I\) – set of all “4U” products (identified by their *Product Name* in the filtered table).  
* \(r_i\) – revenue per unit of product \(i\).  
* \(d_i\) – demand quantity of product \(i\).  
* \(inv_i\) – initial inventory of product \(i\).  
* \(x_i\) – number of units of product \(i\) to sell (decision variable).

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le d_i &&\forall i\in I \\[2pt]
            & 0 \le x_i \le inv_i &&\forall i\in I \\[2pt]
            & x_i \in \mathbb{Z}_+ &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(I\) | `file_0_view_0` | All rows after the filter (`Product Name` contains “4U”). |
| \(r_i\) | `file_0_view_0` | `Revenue` column, indexed by `Product Name`. |
| \(d_i\) | `file_0_view_0` | `Demand` column, indexed by `Product Name`. |
| \(inv_i\) | `file_0_view_0` | `Initial Inventory` column, indexed by `Product Name`. |
| \(x_i\) | – | Decision variable, integer, non‑negative. |

*Table ID:* `file_0_view_0`  
*Columns used:* `Product Name`, `Revenue`, `Demand`, `Initial Inventory`  

The mapping preserves the exact identifiers and values from the source table; no numerical values are enumerated in the model.