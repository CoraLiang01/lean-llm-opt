**Mathematical Model**

Let  

* \(I\) – set of products (indexed by the values in the column **Product Name**).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (column **Revenue**).  
  * \(d_i\) – demand quantity (column **Demand**).  
  * \(s_i\) – initial inventory (column **Initial Inventory**).  

Decision variable  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

---

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

---

**Constraints**

\[
\begin{aligned}
x_i &\le d_i &&\forall i\in I \quad\text{(cannot exceed demand)}\\[4pt]
x_i &\le s_i &&\forall i\in I \quad\text{(cannot exceed available inventory)}\\[4pt]
x_i &\ge 0 &&\forall i\in I
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

All parameters are taken directly from the supplied CSV rows; no additional data are introduced.