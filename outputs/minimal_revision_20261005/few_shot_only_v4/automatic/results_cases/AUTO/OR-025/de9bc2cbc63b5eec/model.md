#### Abstract Mathematical Model

**Index Sets:**

- $I$ : Set of all ‘TABLET’ smartphone models.

**Parameters:**

- $A_i$ : Revenue per unit of model $i \in I$ (from column ‘Revenue’).
- $d_i$ : Total demand for model $i \in I$ (from column ‘Demand’).
- $I_i$ : Initial inventory for model $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**

- $x_i$ : Number of units of model $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

**Objective:**

$$
\max \sum_{i \in I} A_i \, x_i
$$

**Constraints:**

1. **Inventory Constraints:**
   $$
   x_i \leq I_i \qquad \forall i \in I
   $$
2. **Demand Constraints:**
   $$
   x_i \leq d_i \qquad \forall i \in I
   $$
3. **Non-negativity and Integrality:**
   $$
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   $$

---

#### Data Mapping

- **Source Table:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM16/SmartphoneRetailOutletSalesData.csv`
- **Index Set $I$:** All rows where `Product Name` begins with `'TABLET_'`
- **Parameter $A_i$:** Column `Revenue`
- **Parameter $d_i$:** Column `Demand`
- **Parameter $I_i$:** Column `Initial Inventory`