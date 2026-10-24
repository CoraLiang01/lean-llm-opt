**Mathematical Model**

Let  

- \(I\) be the set of all products whose *Product Name* contains the string “Baby” (as returned by the query).  
- For each \(i\in I\):

| Symbol | Meaning | Source column |
|--------|---------|---------------|
| \(R_i\) | Revenue per unit | `Revenue` |
| \(D_i\) | Total demand over the horizon | `Demand` |
| \(I_i\) | Initial inventory | `Initial Inventory` |

Decision variable  

- \(x_i \in \mathbb{Z}_{+}\) – number of units of product \(i\) to be fulfilled.

---

**Objective**

\[
\max \;\; \sum_{i\in I} R_i\,x_i
\]

---

**Constraints**

1. **Demand limits**  
   \[
   x_i \;\le\; D_i , \qquad \forall i\in I
   \]

2. **Inventory limits**  
   \[
   x_i \;\le\; I_i , \qquad \forall i\in I
   \]

3. **Non‑negativity / integrality**  
   \[
   x_i \in \mathbb{Z}_{+}, \qquad \forall i\in I
   \]

---

**Data Mapping**

| Parameter | Table ID | Column name |
|-----------|----------|-------------|
| \(R_i\) | `file_0_view_0` | `Revenue` |
| \(D_i\) | `file_0_view_0` | `Demand` |
| \(I_i\) | `file_0_view_0` | `Initial Inventory` |
| Index \(i\) | `file_0_view_0` | `Product Name` |

*No numeric values are listed; the model references the data via the table and column identifiers above.*