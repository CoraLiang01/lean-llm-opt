**Mathematical model**

Let  

* \(I\) – set of products (indexed by the values in the *Product Name* column).  
* \(r_i\) – revenue per unit of product \(i\) (from *Revenue*).  
* \(d_i\) – demand of product \(i\) (from *Demand*).  
* \(s_i\) – initial inventory of product \(i\) (from *Initial Inventory*).  
* \(x_i\) – number of units of product \(i\) to be sold (decision variable).

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le d_i &&\forall i\in I \quad\text{(demand)}\\
& 0 \le x_i \le s_i &&\forall i\in I \quad\text{(inventory)}\\
& x_i \in \mathbb{R}_{+} &&\forall i\in I
\end{aligned}
\]

---

**Data‑mapping rules**

| Parameter | Source table | Source column | Mapping rule |
|-----------|--------------|---------------|--------------|
| \(I\) | `file_0_view_0` | `Product Name` | Each distinct value forms an index element. |
| \(r_i\) | `file_0_view_0` | `Revenue` | `r_i = value in Revenue column for product i`. |
| \(d_i\) | `file_0_view_0` | `Demand` | `d_i = value in Demand column for product i`. |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` | `s_i = value in Initial Inventory column for product i`. |

No additional constraints or parameters are required beyond those derived from the supplied table.