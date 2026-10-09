---

### Abstract Mathematical Model

#### Index Sets
- $I$: Set of all baked goods (products) in the bakery.

#### Parameters
- $A_i$: Revenue per unit of product $i \in I$.
- $d_i$: Total deterministic demand for product $i \in I$ over the sales horizon.
- $I_i$: Initial inventory available for product $i \in I$.

#### Decision Variables
- $x_i$: Quantity of product $i \in I$ to fulfill (integer, $x_i \geq 0$).

#### Objective
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints
1. **Inventory and Demand Fulfillment Bounds:**
   \[
   0 \leq x_i \leq \min\{I_i, d_i\} \quad \forall i \in I
   \]
   (Or equivalently, two separate constraints:)
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
2. **Variable Domain:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

### Data Mapping

- **Table:** `file_0_view_0` (from `Frenchbakerydailysales.csv`)
- **Index Set $I$:** All records in column `Product Name`
- **Parameter $A_i$:** Column `Revenue`
- **Parameter $d_i$:** Column `Demand`
- **Parameter $I_i$:** Column `Initial Inventory`
- **No filters**: All records from the table are included.

---