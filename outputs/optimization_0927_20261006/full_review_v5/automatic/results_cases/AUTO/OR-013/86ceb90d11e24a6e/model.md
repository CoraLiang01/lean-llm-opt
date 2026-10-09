#### Abstract Mathematical Model

**Index Sets**

- $I$: Set of “4U” products (indexed by $i$).

**Parameters**

- $A_i$: Revenue per unit of product $i$.
- $d_i$: Demand for product $i$ over the sales horizon.
- $I_i$: Initial inventory of product $i$.

**Decision Variables**

- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$.

**Objective**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints**

1. **Inventory Constraints:** 
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$
2. **Demand Constraints:** 
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
3. **Non-negativity and Integrality:** 
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Source Table:** `file_0_view_0` (from `OnlineSalesinUSA.csv`)
- **Index Set $I$:** All records where `Product Name` has prefix "4U"
- **Parameter $A_i$:** Column `Revenue`
- **Parameter $d_i$:** Column `Demand`
- **Parameter $I_i$:** Column `Initial Inventory`