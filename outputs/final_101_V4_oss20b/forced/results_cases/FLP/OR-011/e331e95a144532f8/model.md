**Mathematical Model**

Let  

* \(I\) be the set of product identifiers whose `id_number` starts with **id999**.  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**)  
  * \(d_i\) – demand during the horizon (from column **Demand**)  
  * \(v_i\) – initial inventory available (from column **Initial Inventory**)  

Decision variable  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad\text{(units of product }i\text{ to fulfill)}
\]

---

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

---

**Constraints**

1. **Demand satisfaction**  
   \[
   x_i \;\le\; d_i \qquad \forall i\in I
   \]

2. **Inventory limitation**  
   \[
   x_i \;\le\; v_i \qquad \forall i\in I
   \]

3. **Non‑negativity & integrality**  
   \[
   x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
   \]

---

**Data Mapping**

| Symbol | Source | Table ID | Column | Filter |
|--------|--------|----------|--------|--------|
| \(I\) | Index set | `file_0_view_0` | `id_number` | `id_number` starts with `"id999"` |
| \(r_i\) | Revenue | `file_0_view_0` | `Revenue` | – |
| \(d_i\) | Demand | `file_0_view_0` | `Demand` | – |
| \(v_i\) | Initial Inventory | `file_0_view_0` | `Initial Inventory` | – |

*The table `file_0_view_0` is the result of filtering `OnlineRetailSalesDataset.csv` for rows where `id_number` has the prefix `"id999"`.*