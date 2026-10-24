**Mathematical Model**

Let  

* \(I\) – set of products (identified by the *Product Name* column).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from *Revenue*).  
  * \(d_i\) – demand (from *Demand*).  
  * \(s_i\) – initial inventory (from *Initial Inventory*).  

Decision variable  

* \(x_i\ge 0\) – number of units of product \(i\) that will be sold (fulfilled).

\[
\begin{aligned}
\max\;&\sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;&x_i \;\le\; d_i &&\forall i\in I \\[2pt]
            &x_i \;\le\; s_i &&\forall i\in I \\[2pt]
            &x_i \;\ge\; 0 &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(I\) | Index set | `file_0_view_0` | `Product Name` |
| \(r_i\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand | `file_0_view_0` | `Demand` |
| \(s_i\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |

All numeric columns (`Revenue`, `Demand`, `Initial Inventory`) are parsed as floating‑point numbers.  
The mapping uses the complete runtime records; no values are enumerated in the model.