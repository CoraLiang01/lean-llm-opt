**Mathematical Model**

Let  

* \(I\) – set of all products whose *Product Name* contains “4U” (extracted from the
  filtered view `file_0_view_0`).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – demand (from column **Demand**).  
  * \(s_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad\forall i\in I
\]

*\(x_i\)* = number of units of product \(i\) that the retailer chooses to sell.

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

**Constraints**

1. **Demand limit**  
   \[
   x_i \;\le\; d_i \qquad\forall i\in I
   \]

2. **Inventory limit**  
   \[
   x_i \;\le\; s_i \qquad\forall i\in I
   \]

3. **Non‑negativity & integrality**  
   \[
   x_i \in \mathbb{Z}_{\ge 0}\qquad\forall i\in I
   \]

---

**Data Mapping Rules**

| Index / Parameter | Source Table | Source Column | Mapping Description |
|-------------------|--------------|---------------|----------------------|
| \(I\) | `file_0_view_0` | `Product Name` | Each distinct product name containing “4U” becomes an index element. |
| \(r_i\) | `file_0_view_0` | `Revenue` | Revenue per unit for product \(i\). |
| \(d_i\) | `file_0_view_0` | `Demand` | Known demand for product \(i\). |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` | Initial stock available for product \(i\). |

These mappings directly bind the symbolic parameters to the corresponding columns in the filtered view, ensuring that every identifier and coefficient used in the model originates from the supplied data.