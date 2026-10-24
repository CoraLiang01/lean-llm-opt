**Mathematical Model**

Let  

* \(I=\{1,\dots,5\}\) be the set of books that satisfy the filter  
  (rows returned from *DifferentStoreSales.csv* where `Product_Name` starts with “Books”).

Parameters (obtained from the filtered table)

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(R_i\) | Revenue per unit of book \(i\) | `Revenue` column |
| \(D_i\) | Demand quantity of book \(i\) | `Demand` column |
| \(S_i\) | Initial inventory of book \(i\) | `Initial Inventory` column |

Decision variables  

* \(x_i\) – number of units of book \(i\) to be sold (fulfilled).  
  \(x_i \in \mathbb{Z}_{+}\).

Objective  

\[
\max \;\; \sum_{i\in I} R_i\,x_i
\]

Constraints  

1. **Demand limitation**  
   \[
   0 \;\le\; x_i \;\le\; D_i , \qquad \forall i\in I
   \]

2. **Inventory limitation**  
   \[
   0 \;\le\; x_i \;\le\; S_i , \qquad \forall i\in I
   \]

(If \(S_i < D_i\) the inventory constraint will be binding; otherwise the demand constraint will be.)

---

**Data‑Mapping Rules**

| Index set | Definition | Source |
|-----------|------------|--------|
| \(I\) | All rows in *DifferentStoreSales.csv* where `Product_Name` has prefix “Books” | `filters` in the observation |
| \(R_i\) | Value of `Revenue` for row \(i\) | `Revenue` column |
| \(D_i\) | Value of `Demand` for row \(i\) | `Demand` column |
| \(S_i\) | Value of `Initial Inventory` for row \(i\) | `Initial Inventory` column |

No other parameters or variables are introduced. The model uses only the data returned by the filter and respects the variable domains and constraints specified in the query.