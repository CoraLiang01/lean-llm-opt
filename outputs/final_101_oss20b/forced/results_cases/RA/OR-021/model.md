**Mathematical Model**

Let  

- \(I\) be the set of all products (identified by *Product Name*).  
- For each \(i\in I\)  
  - \(r_i\)  – revenue per unit (column **Revenue**)  
  - \(d_i\)  – demand quantity (column **Demand**)  
  - \(s_i\)  – initial inventory (column **Initial Inventory**)  

Decision variable  

- \(x_i\in\mathbb{Z}_{\ge 0}\) – number of units of product \(i\) to fulfill.

\[
\begin{aligned}
\max\;&\sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;&0 \le x_i \le d_i &&\forall i\in I \\[2pt]
&x_i \le s_i &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | `file_0_view_0` | `Product Name` |
| \(r_i\) | `file_0_view_0` | `Revenue` |
| \(d_i\) | `file_0_view_0` | `Demand` |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` |
| \(x_i\) | – | – |

All parameters are taken directly from the complete runtime records of `Salesofsummerclothes.csv` (table id `file_0_view_0`). No values are enumerated in the model; the solver will read the full dataset.