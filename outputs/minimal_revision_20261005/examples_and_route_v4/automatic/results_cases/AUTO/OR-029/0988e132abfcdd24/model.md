**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of all products in ZARASales.csv classified under ‘FAUX’ (i.e., all products whose name contains "FAUX").
  
**Parameters:**
- $r_i$: Revenue per unit of product $i$ (from column ‘Revenue’ in ZARASales.csv).
- $d_i$: Demand for product $i$ (from column ‘Demand’ in ZARASales.csv).
- $s_i$: Initial inventory of product $i$ (from column ‘Initial Inventory’ in ZARASales.csv).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$.

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Inventory constraint:** 
   \[
   x_i \leq s_i \qquad \forall i \in I
   \]
2. **Demand constraint:** 
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]
3. **Nonnegativity and integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
   \]

---

**Data Mapping**

- $I$: All rows in `file_0_view_0` (ZARASales.csv) where `Product Name` contains "FAUX".
- $r_i$: `Revenue` column in `file_0_view_0`, indexed by `Product Name`.
- $d_i$: `Demand` column in `file_0_view_0`, indexed by `Product Name`.
- $s_i$: `Initial Inventory` column in `file_0_view_0`, indexed by `Product Name`.
- $x_i$: Decision variable for each $i \in I$.

**Notes:**  
- All parameters and index sets are defined directly from the returned data, using the exact column and table identifiers.
- No additional constraints or data are invented or omitted.