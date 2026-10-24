## Mathematical Model

**Index set**  
\(I=\{i\mid\) product record \(i\) returned from the filtered view of *OnlineSalesinUSA.csv*\(\}\)

**Parameters** (all taken from the filtered view)

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(r_i\) | Revenue per unit of product \(i\) | table_id `file_0_view_0`, column `Revenue` |
| \(d_i\) | Demand (units) for product \(i\) | table_id `file_0_view_0`, column `Demand` |
| \(I_i\) | Initial inventory of product \(i\) | table_id `file_0_view_0`, column `Initial Inventory` |

**Decision variables**

\[
x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\]

\(x_i\) = number of units of product \(i\) to sell.

**Objective**

\[
\max \;\; \sum_{i\in I} r_i\,x_i
\]

**Constraints**

1. **Demand limit**  
   \[
   x_i \;\le\; d_i \qquad \forall i\in I
   \]

2. **Inventory limit**  
   \[
   x_i \;\le\; I_i \qquad \forall i\in I
   \]

No other constraints are required because restocking is not allowed and there is no in‑transit inventory.

---

## Data Mapping Rules

| Index | Source | Column | Filter |
|-------|--------|--------|--------|
| \(i\) | `file_0_view_0` | `Product Name` | `contains "4U"` (case‑sensitive) |
| \(r_i\) | `file_0_view_0` | `Revenue` | – |
| \(d_i\) | `file_0_view_0` | `Demand` | – |
| \(I_i\) | `file_0_view_0` | `Initial Inventory` | – |

The index set \(I\) is defined as the set of all rows that satisfy the filter condition on `Product Name`. Each parameter is mapped directly to the corresponding column in the filtered view; no additional transformations are applied. The variable domain and constraints are defined solely in terms of these mapped parameters.