**Mathematical model**

Let  

* \(I\) – set of products (identified by *Product Name*).  
* \(r_i\) – revenue per unit of product \(i\) (from *Revenue*).  
* \(d_i\) – demand for product \(i\) (from *Demand*).  
* \(s_i\) – initial inventory of product \(i\) (from *Initial Inventory*).  
* \(x_i\) – quantity of product \(i\) to be sold (decision variable).

\[
\begin{aligned}
\max\;&\sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le d_i &&\forall i\in I \\[2pt]
& 0 \le x_i \le s_i &&\forall i\in I
\end{aligned}
\]

All variables are continuous non‑negative.

---

**Data‑mapping rules**

| Parameter | Source column | Source table |
|-----------|---------------|--------------|
| \(I\) (index set) | *Product Name* | `file_0_view_0` |
| \(r_i\) | *Revenue* | `file_0_view_0` |
| \(d_i\) | *Demand* | `file_0_view_0` |
| \(s_i\) | *Initial Inventory* | `file_0_view_0` |

The table `file_0_view_0` contains 148 rows, each row providing the values for one product. No additional aggregation or transformation is required.