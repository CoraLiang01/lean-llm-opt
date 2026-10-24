**Mathematical Model**

Let  

* \(I\) be the set of product identifiers whose `id_number` starts with **id999**.  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**)  
  * \(d_i\) – demand during the horizon (from column **Demand**)  
  * \(s_i\) – initial inventory (from column **Initial Inventory**)  

Decision variable  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

Objective (maximize total revenue)

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

Constraints  

1. **Demand satisfaction**  
   \[
   x_i \;\le\; d_i \qquad \forall i\in I
   \]

2. **Inventory limitation**  
   \[
   x_i \;\le\; s_i \qquad \forall i\in I
   \]

3. **Non‑negativity & integrality**  
   \[
   x_i \;\ge\; 0,\; x_i \in \mathbb{Z}\qquad \forall i\in I
   \]

---

**Data Mapping Rules**

| Parameter / Variable | Source Table | Source Column | Notes |
|----------------------|--------------|---------------|-------|
| \(I\) | `file_0_view_0` | `id_number` | Include rows where `id_number` starts with `id999` |
| \(r_i\) | `file_0_view_0` | `Revenue` | Revenue per unit for product \(i\) |
| \(d_i\) | `file_0_view_0` | `Demand` | Known demand for product \(i\) |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` | Initial stock available for product \(i\) |
| \(x_i\) | – | – | Decision variable, integer, non‑negative |

All parameters are taken directly from the filtered view `file_0_view_0`; no additional data or aggregation is introduced. The model is ready for numerical instantiation with the single product record returned in the observation.