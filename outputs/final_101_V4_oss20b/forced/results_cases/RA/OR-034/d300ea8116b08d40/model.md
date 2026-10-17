**Mathematical Model**

Let  

* \(I\) be the set of all products (one element per row of *Frenchbakerydailysales.csv*).  
* For each \(i\in I\) let  

  * \(r_i\)  – revenue per unit (column **Revenue**).  
  * \(d_i\)  – demand (column **Demand**).  
  * \(s_i\)  – initial inventory (column **Initial Inventory**).  

Decision variable  

* \(x_i\in\mathbb{Z}_{\ge 0}\) – quantity of product \(i\) to be sold.

---

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& x_i \;\le\; s_i &&\forall i\in I \\[2pt]
& x_i \;\le\; d_i &&\forall i\in I \\[2pt]
& x_i \;\ge\; 0 &&\forall i\in I \\[2pt]
& x_i \in \mathbb{Z} &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column Name |
|-----------|--------------|-------------|
| \(r_i\)   | `file_0_view_0` | `Revenue` |
| \(d_i\)   | `file_0_view_0` | `Demand` |
| \(s_i\)   | `file_0_view_0` | `Initial Inventory` |
| Index set \(I\) | `file_0_view_0` | `Product Name` |

All indices and parameter values are taken directly from the complete runtime records of the table `file_0_view_0`. No values are enumerated or hard‑coded in the model.