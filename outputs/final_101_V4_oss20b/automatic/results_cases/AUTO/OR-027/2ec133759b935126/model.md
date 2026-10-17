**Mathematical Model**

Let  

* \(I\) – index set of all products whose *Sub Category* contains the word “Organ”.  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – total demand over the sales horizon (from column **Demand**).  
  * \(I_i\) – initial inventory available (from column **Initial Inventory**).  

Decision variable  

* \(x_i\) – number of units of product \(i\) to be sold (integer, \(\ge 0\)).

---

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\
\text{s.t. } & 0 \le x_i \le d_i && \forall i\in I \\
             & 0 \le x_i \le I_i && \forall i\in I \\
             & x_i \in \mathbb{Z} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(r_i\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_i\) | Total demand | `file_0_view_0` | `Demand` |
| \(I_i\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |
| Index set \(I\) | Product identifiers | `file_0_view_0` | `Sub Category` (filtered where contains “Organ”) |

All parameter values are taken directly from the corresponding columns of the filtered view `file_0_view_0`. No additional data or assumptions are introduced.