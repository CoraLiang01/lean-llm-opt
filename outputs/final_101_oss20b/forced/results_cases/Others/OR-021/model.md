**Mathematical Model**

Let  

* \(I=\{1,\dots,68\}\) be the set of products (one index per row of *Salesofsummerclothes.csv*).  
* For each \(i\in I\)  
  * \(p_i\) – revenue per unit (USD) – *Revenue* column.  
  * \(d_i\) – maximum demand (units) – *Demand* column.  
  * \(s_i\) – initial inventory (units) – *Initial Inventory* column.  

Decision variable  

* \(x_i\in\mathbb{Z}_{\ge 0}\) – number of units of product \(i\) to sell.

\[
\begin{aligned}
\max\;&\sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t.}\;&x_i \le d_i &&\forall i\in I \quad\text{(demand limit)}\\
            &x_i \le s_i &&\forall i\in I \quad\text{(inventory limit)}\\
            &x_i\in\mathbb{Z}_{\ge 0} &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Column | Notes |
|-----------|--------|--------|-------|
| \(p_i\) | *Salesofsummerclothes.csv* | *Revenue* | revenue per unit (float) |
| \(d_i\) | *Salesofsummerclothes.csv* | *Demand* | maximum demand (integer) |
| \(s_i\) | *Salesofsummerclothes.csv* | *Initial Inventory* | available inventory (integer) |
| \(I\) | *Salesofsummerclothes.csv* | *Product Name* | each distinct product row defines an index |

All parameters are taken directly from the corresponding columns of the single source table *file_0_view_0*. No additional data or derived parameters are introduced. The model uses only the constraints and objective specified in the query.