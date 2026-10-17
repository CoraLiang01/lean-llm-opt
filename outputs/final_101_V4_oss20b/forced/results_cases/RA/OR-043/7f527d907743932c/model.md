**Mathematical Model**

Let  

* \(I\) – set of drug products (indexed by `ProductName` from *products.csv*).  
* \(x_i\) – number of units of product \(i\) ordered each day.  

Parameters  

* \(v_i\) – benefit per unit of product \(i\) (`Value` column).  
* \(w_i\) – weight (space) per unit of product \(i\) (`Weight` column).  
* \(C\) – total stock capacity (`Capacity` column).  

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | `file_1_view_0` | `ProductName` |
| \(v_i\) | `file_1_view_0` | `Value` |
| \(w_i\) | `file_1_view_0` | `Weight` |
| \(C\) | `file_0_view_0` | `Capacity` |

All parameters are taken directly from the corresponding CSV files; no values are hard‑coded.