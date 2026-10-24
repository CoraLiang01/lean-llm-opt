**Mathematical Model**

Let  

* \(I=\{\text{Organic Fruits},\text{Organic Staples},\text{Organic Vegetables}\}\) – the set of “Organ” products that appear in the filtered view.  

* For each \(i\in I\)  
  * \(R_i\) – revenue per unit (column **Revenue**).  
  * \(D_i\) – demand (column **Demand**).  
  * \(I_i\) – initial inventory (column **Initial Inventory**).  

* Decision variable  
  * \(x_i\in \mathbb{Z}_{+}\) – number of units of product \(i\) that will be sold (fulfilled).

---

**Objective**

\[
\max \;\; \sum_{i\in I} R_i\,x_i
\]

---

**Constraints**

1. **Demand limit**  
   \[
   x_i \;\le\; D_i \qquad \forall i\in I
   \]

2. **Inventory limit**  
   \[
   x_i \;\le\; I_i \qquad \forall i\in I
   \]

3. **Non‑negativity**  
   \[
   x_i \;\ge\; 0 \qquad \forall i\in I
   \]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(R_i\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(D_i\) | Demand quantity | `file_0_view_0` | `Demand` |
| \(I_i\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |
| Index set \(I\) | Rows where `Sub Category` contains `"Organ"` | `file_0_view_0` | `Sub Category` |

The filtered view `file_0_view_0` contains exactly the three rows shown in the observation, so the index set \(I\) consists of those three product names. No other rows are included because the filter condition `Sub Category contains "Organ"` is applied.