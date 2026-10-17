**Mathematical Model**

Let  

- \(I\) – index set of all products whose *Product Name* contains the word *Fashion* (as filtered in the query).  

Parameters (all taken directly from the source table `file_0_view_0`):  

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(r_i\) | Revenue per unit of product \(i\) | `Revenue` column |
| \(d_i\) | Demand (known, deterministic) for product \(i\) | `Demand` column |
| \(s_i\) | Initial inventory of product \(i\) | `Initial Inventory` column |

Decision variables  

- \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) to be sold (fulfilled).

**Objective**

\[
\max \;\; \sum_{i\in I} r_i\,x_i
\]

**Constraints**

1. **Demand limitation**  
   \[
   x_i \;\le\; d_i , \qquad \forall i\in I
   \]

2. **Inventory limitation**  
   \[
   x_i \;\le\; s_i , \qquad \forall i\in I
   \]

3. **Non‑negativity / integrality**  
   \[
   x_i \in \mathbb{Z}_{\ge 0}, \qquad \forall i\in I
   \]

---

**Data Mapping**

| Index / Parameter | Source Table | Source Column |
|-------------------|--------------|---------------|
| \(I\) | `file_0_view_0` | – (derived from all rows after the *Fashion* filter) |
| \(r_i\) | `file_0_view_0` | `Revenue` |
| \(d_i\) | `file_0_view_0` | `Demand` |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` |

No additional data or constraints are introduced beyond those explicitly stated in the query.