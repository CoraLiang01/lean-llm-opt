**Mathematical Model**

Let  

* \(I\) be the set of product records that satisfy the filter  
  `id_number` prefix `id999` (here \(I=\{i\}\)).  

Parameters (all taken from the view **file_0_view_0**):

| Symbol | Meaning | Source column |
|--------|---------|---------------|
| \(rev_i\) | Revenue per unit of product \(i\) | `Revenue` |
| \(dem_i\) | Demand for product \(i\) during the horizon | `Demand` |
| \(inv_i\) | Initial inventory of product \(i\) | `Initial Inventory` |

Decision variables:

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

Objective (maximize total revenue):

\[
\max \;\sum_{i\in I} rev_i \; x_i
\]

Constraints:

1. **Demand limit**  
   \[
   x_i \;\le\; dem_i \qquad \forall i\in I
   \]

2. **Inventory limit**  
   \[
   x_i \;\le\; inv_i \qquad \forall i\in I
   \]

3. **Non‑negativity** (already encoded in variable domain).

---

**Data Mapping**

| Parameter | Table ID | Column name |
|-----------|----------|-------------|
| \(rev_i\) | file_0_view_0 | Revenue |
| \(dem_i\) | file_0_view_0 | Demand |
| \(inv_i\) | file_0_view_0 | Initial Inventory |

All indices, parameters, and variables are defined symbolically; no numerical values are hard‑coded.