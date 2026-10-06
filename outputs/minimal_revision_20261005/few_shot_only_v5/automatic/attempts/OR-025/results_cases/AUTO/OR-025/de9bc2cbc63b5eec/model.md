#### Abstract Mathematical Model

**Index Sets:**

- $I$ : Set of all ‘TABLET’ smartphone models (indexed by $i$).

**Parameters:**

- $A_i$ : Revenue per unit for model $i$ (from column ‘Revenue’).
- $d_i$ : Deterministic demand for model $i$ (from column ‘Demand’).
- $I_i$ : Initial inventory for model $i$ (from column ‘Initial Inventory’).

**Decision Variables:**

- $x_i$ : Number of units of model $i$ to fulfill, $\forall i \in I$.

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

- **Source Table:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM16/SmartphoneRetailOutletSalesData.csv`
- **Index Set:** All records where `Product Name` begins with `'TABLET_'`
- **Parameters:**
  - $A_i$: `Revenue`
  - $d_i$: `Demand`
  - $I_i$: `Initial Inventory`