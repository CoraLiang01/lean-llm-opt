#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of all products where `Product Name` identifies a 'TABLET' smartphone model (from `file_0_view_0`).

**Parameters:**
- $A_i$: Revenue per unit for product $i \in I$ (from column `Revenue` in `file_0_view_0`).
- $d_i$: Deterministic demand for product $i \in I$ (from column `Demand` in `file_0_view_0`).
- $I_i$: Initial inventory for product $i \in I$ (from column `Initial Inventory` in `file_0_view_0`).

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:**  
   \[
   x_i \leq I_i \qquad \forall i \in I
   \]
2. **Demand Constraint:**  
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]
3. **Non-negativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   \]

---

#### Data Mapping

- **Source Table:** `file_0_view_0` (from `SmartphoneRetailOutletSalesData.csv`)
- **Index Set $I$:** All rows where `Product Name` identifies a 'TABLET' smartphone model (FALLBACK_FULL_DATA: selection is explicit, as filter evidence is not present in the query).
- **Parameter $A_i$:** Column `Revenue`
- **Parameter $d_i$:** Column `Demand`
- **Parameter $I_i$:** Column `Initial Inventory`
- **Decision Variable $x_i$:** Number of units fulfilled for each $i \in I$.

No additional constraints or subsets are imposed by the query. All 'TABLET' smartphone models in the data are included.