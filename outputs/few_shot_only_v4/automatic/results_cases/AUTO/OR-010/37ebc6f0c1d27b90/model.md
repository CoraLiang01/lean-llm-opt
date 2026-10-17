#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of all products (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit for product $i$ (from column "Revenue" in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM1/MobileSalesDataset.csv).
- $d_i$: Expected demand for product $i$ during the sales cycle (from column "Demand").
- $I_i$: Initial inventory for product $i$ (from column "Initial Inventory").

**Decision Variables:**
- $x_i$: Number of orders fulfilled for product $i$; $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. **Demand Constraint:** 
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Source file:** /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM1/MobileSalesDataset.csv
- **Index set $I$:** All rows in the file, each corresponding to a unique product (column "Product Name").
- **Parameter $A_i$:** Value from column "Revenue" for product $i$.
- **Parameter $d_i$:** Value from column "Demand" for product $i$.
- **Parameter $I_i$:** Value from column "Initial Inventory" for product $i$.
- **Decision variable $x_i$:** Fulfilled quantity for product $i$.

All columns and rows from the file are included; no subset or filter is applied.