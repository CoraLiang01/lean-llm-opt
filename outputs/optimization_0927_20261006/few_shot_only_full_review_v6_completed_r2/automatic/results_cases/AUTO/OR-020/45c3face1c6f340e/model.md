---

### Abstract Mathematical Model

#### Index Sets
- $I$: Set of all products, indexed by $i$.

#### Parameters
- $A_i$: Revenue per unit of product $i$.  
- $d_i$: Demand for product $i$.  
- $I_i$: Initial inventory for product $i$.

#### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

#### Objective
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints
1. **Inventory and Demand Bounds**  
   \[
   0 \leq x_i \leq \min\{d_i, I_i\}, \quad \forall i \in I
   \]
   (Equivalently, $x_i \leq d_i$ and $x_i \leq I_i$ for all $i$.)

2. **Variable Domain**  
   \[
   x_i \in \mathbb{R}_+, \quad \forall i \in I
   \]
   (If integer fulfillment is required, replace $\mathbb{R}_+$ with $\mathbb{Z}_+$.)

---

### Data Mapping

- **Table:** `file_0_view_0` (from `SalesDatainBusinesses.csv`)
- **Index Set $I$:** All records in column `Product Name`
- **Parameter $A_i$:** Column `Revenue`
- **Parameter $d_i$:** Column `Demand`
- **Parameter $I_i$:** Column `Initial Inventory`
- **No filters**: All rows are included; no subset or eligibility condition is applied.

---