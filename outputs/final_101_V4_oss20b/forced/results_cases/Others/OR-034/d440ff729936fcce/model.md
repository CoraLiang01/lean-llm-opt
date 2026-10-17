## Mathematical Model

**Indices**  
\(i \in \mathcal{I}\) – set of all baked‑good products.

**Parameters** (from table `file_0_view_0`)  

| Parameter | Symbol | Source column |
|-----------|--------|---------------|
| Revenue per unit | \(r_i\) | `Revenue` |
| Demand (units) | \(d_i\) | `Demand` |
| Initial inventory (units) | \(s_i\) | `Initial Inventory` |

**Decision variables**  

\(x_i \ge 0\) – quantity of product \(i\) that the bakery plans to sell.

**Objective**  

\[
\max \;\sum_{i\in\mathcal{I}} r_i\,x_i
\]

**Constraints**

1. **Demand limit**  
   \[
   x_i \;\le\; d_i \qquad \forall i\in\mathcal{I}
   \]

2. **Inventory limit**  
   \[
   x_i \;\le\; s_i \qquad \forall i\in\mathcal{I}
   \]

3. **Non‑negativity**  
   \[
   x_i \;\ge\; 0 \qquad \forall i\in\mathcal{I}
   \]

---

## Data Mapping Rules

| Table ID | Column | Parameter / Variable |
|----------|--------|----------------------|
| `file_0_view_0` | `Product Name` | Index set \(\mathcal{I}\) (product identifiers) |
| `file_0_view_0` | `Revenue` | \(r_i\) |
| `file_0_view_0` | `Demand` | \(d_i\) |
| `file_0_view_0` | `Initial Inventory` | \(s_i\) |

All parameters are taken directly from the corresponding columns of the complete dataset (148 rows). No additional filtering or aggregation is applied beyond the constraints listed above.