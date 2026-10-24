**Mathematical Model**

Let  

* \(I\) – set of product identifiers (one per row of *RetailSalesDataset.csv*).  
* For each \(i\in I\)  
  * \(d_i\) – demand of product \(i\) (column **Demand**).  
  * \(s_i\) – initial inventory of product \(i\) (column **Initial Inventory**).  
  * \(r_i\) – revenue per unit of product \(i\) (column **Revenue**).  

Decision variable  

\[
x_i \;\ge\; 0 \qquad \forall i\in I
\]

*\(x_i\)* = quantity of product \(i\) to be sold (fulfilled).

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

**Constraints**

1. **Demand limit**  
   \[
   x_i \;\le\; d_i \qquad \forall i\in I
   \]

2. **Inventory limit**  
   \[
   x_i \;\le\; s_i \qquad \forall i\in I
   \]

3. **Non‑negativity** (already stated in variable definition).

---

**Data Mapping Rules**

| Parameter | Source Table | Column Name |
|-----------|--------------|-------------|
| \(I\) | `file_0_view_0` | `Product Name` |
| \(d_i\) | `file_0_view_0` | `Demand` |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` |
| \(r_i\) | `file_0_view_0` | `Revenue` |

*All values are taken directly from the corresponding columns of the single source table `file_0_view_0`.*