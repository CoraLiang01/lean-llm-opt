**Mathematical Model**

Let  

* \(I\) – set of products (indexed by the *Product Name* values).  

Parameters (all obtained from table **file_0_view_0**):  

| Symbol | Meaning | Source column |
|--------|---------|---------------|
| \(r_i\) | revenue per unit of product \(i\) | `Revenue` |
| \(d_i\) | demand quantity of product \(i\) | `Demand` |
| \(inv_i\) | initial inventory of product \(i\) | `Initial Inventory` |

Decision variable  

* \(x_i \ge 0\) – number of units of product \(i\) to sell (continuous).

Objective  

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

Constraints  

1. **Demand satisfaction**  
   \[
   x_i \;\le\; d_i , \qquad \forall i\in I
   \]

2. **Inventory availability**  
   \[
   x_i \;\le\; inv_i , \qquad \forall i\in I
   \]

3. **Non‑negativity**  
   \[
   x_i \;\ge\; 0 , \qquad \forall i\in I
   \]

---

**Data Mapping**

| Index set | Table ID | Column mapping |
|-----------|----------|----------------|
| \(I\) | `file_0_view_0` | `Product Name` → product identifier |
| Parameters | `file_0_view_0` | `Revenue` → \(r_i\) |
| | | `Demand` → \(d_i\) |
| | | `Initial Inventory` → \(inv_i\) |

No numerical values are enumerated; all coefficients are linked directly to the source columns.