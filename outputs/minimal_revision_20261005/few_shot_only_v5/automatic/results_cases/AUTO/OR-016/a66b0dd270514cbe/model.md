#### Abstract Mathematical Model

**Index Sets:**

- $I$ : Set of all product categories (indexed by $i$).

**Parameters:**

- $A_i$ : Revenue per unit for product $i$.
- $d_i$ : Demand for product $i$.
- $I_i$ : Initial inventory for product $i$.

**Decision Variables:**

- $x_i$ : Number of units of product $i$ to fulfill (integer, $x_i \geq 0$).

**Objective:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Demand fulfillment constraint:**
   $$
   x_i \leq d_i \quad \forall i \in I
   $$

2. **Inventory constraint:**
   $$
   x_i \leq I_i \quad \forall i \in I
   $$

3. **Non-negativity and integrality:**
   $$
   x_i \in \mathbb{Z}_{+} \quad \forall i \in I
   $$

---

#### Data Mapping

- **Index set $I$:** All values in column `"Product Name"` of table `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM7/RetailSalesDataset.csv`
- **Parameter $A_i$:** Column `"Revenue"` of the same table
- **Parameter $d_i$:** Column `"Demand"` of the same table
- **Parameter $I_i$:** Column `"Initial Inventory"` of the same table