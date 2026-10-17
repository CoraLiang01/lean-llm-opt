**Mathematical Model**

Let  

* \(I\) – set of all products whose *Product Name* contains the word “Fashion”.  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – deterministic demand (from column **Demand**).  
  * \(s_i\) – initial inventory (from column **Initial Inventory**).  
  * \(x_i\) – number of units of product \(i\) to be sold (decision variable).

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le \min\{d_i,\;s_i\}, \qquad \forall i\in I .
\end{aligned}
\]

The constraint \(x_i \le d_i\) guarantees that demand is not exceeded, while  
\(x_i \le s_i\) ensures that inventory limits are respected.  
All variables are continuous and non‑negative.

---

**Data Mapping**

| Index / Parameter | Source Table | Column(s) | Notes |
|-------------------|--------------|-----------|-------|
| \(I\) | `file_0_view_0` | `Product Name` | All rows where `Product Name` contains “Fashion”. |
| \(r_i\) | `file_0_view_0` | `Revenue` | Revenue per unit for product \(i\). |
| \(d_i\) | `file_0_view_0` | `Demand` | Deterministic demand for product \(i\). |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` | Available inventory for product \(i\). |
| \(x_i\) | Decision variable | – | Number of units of product \(i\) to sell. |

The mapping uses the raw column names exactly as they appear in the CSV view `file_0_view_0`. No additional numeric tables are enumerated; all coefficients are referenced symbolically.