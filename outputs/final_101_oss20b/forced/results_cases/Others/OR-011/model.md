**Mathematical Model**

Let  

* \(I\) – set of products whose `id_number` starts with **id999** (obtained from the filtered view).  
* For each \(i\in I\):  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – demand during the horizon (from column **Demand**).  
  * \(s_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

* \(x_i \in \mathbb{Z}_{\ge 0}\) – units of product \(i\) to fulfill.

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

**Constraints**

\[
\begin{aligned}
x_i &\le d_i &&\forall i\in I &&\text{(cannot exceed demand)}\\
x_i &\le s_i &&\forall i\in I &&\text{(cannot exceed available inventory)}\\
x_i &\ge 0 &&\forall i\in I &&\text{(non‑negativity)}\\
x_i &\in \mathbb{Z} &&\forall i\in I &&\text{(integer units)}
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(r_i\) | Revenue | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand | `file_0_view_0` | `Demand` |
| \(s_i\) | Initial Inventory | `file_0_view_0` | `Initial Inventory` |
| Index set \(I\) | Filtered rows | `file_0_view_0` | `id_number` (prefix `id999`) |

No other data or constraints are required. The model is fully specified by the above equations and mappings.