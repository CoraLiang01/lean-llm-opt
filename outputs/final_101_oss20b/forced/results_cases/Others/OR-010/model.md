**Mathematical Model**

Let  

* \(I\) – set of products (product names from *MobileSalesDataset.csv*).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (column **Revenue**).  
  * \(d_i\) – deterministic demand (column **Demand**).  
  * \(s_i\) – initial inventory (column **Initial Inventory**).  

Decision variable  

* \(x_i\in \mathbb{Z}_{\ge 0}\) – quantity of product \(i\) that is fulfilled.

---

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

---

**Constraints**

1. **Demand limits**  
   \[
   x_i \;\le\; d_i \qquad \forall i\in I
   \]

2. **Inventory limits**  
   \[
   x_i \;\le\; s_i \qquad \forall i\in I
   \]

3. **Non‑negativity & integrality**  
   \[
   x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
   \]

---

**Data Mapping**

| Index / Parameter | Source | Column |
|-------------------|--------|--------|
| \(I\) | *file_0_view_0* | **Product Name** |
| \(r_i\) | *file_0_view_0* | **Revenue** |
| \(d_i\) | *file_0_view_0* | **Demand** |
| \(s_i\) | *file_0_view_0* | **Initial Inventory** |

All parameters are taken directly from the corresponding columns of the table *MobileSalesDataset.csv* (view id *file_0_view_0*). No additional data or transformations are introduced.