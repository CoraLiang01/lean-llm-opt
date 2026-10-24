**Mathematical Model**

Let  

- \(I\) be the set of all products (identified by the *Product Name* column).  
- For each \(i\in I\)  
  - \(rev_i\) = revenue per unit (column *Revenue*).  
  - \(init_i\) = initial inventory (column *Initial Inventory*).  
  - \(dem_i\)  = demand (column *Demand*).  

Decision variable  

- \(x_i\in\mathbb{Z}_{\ge 0}\) = quantity of product \(i\) to sell.

\[
\begin{aligned}
\max\;&\sum_{i\in I} rev_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le init_i &&\forall i\in I \\[2pt]
& 0 \le x_i \le dem_i   &&\forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Table ID          | Column Name          | Parameter / Index |
|-------------------|----------------------|-------------------|
| `file_0_view_0`   | `Product Name`       | index set \(I\)   |
| `file_0_view_0`   | `Revenue`            | \(rev_i\)         |
| `file_0_view_0`   | `Demand`             | \(dem_i\)         |
| `file_0_view_0`   | `Initial Inventory`  | \(init_i\)        |

All parameters are taken directly from the corresponding columns of the table `file_0_view_0`; no additional data are introduced.