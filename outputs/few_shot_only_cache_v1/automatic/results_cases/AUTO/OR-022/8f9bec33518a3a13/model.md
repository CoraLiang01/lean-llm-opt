#### Abstract Mathematical Model

**Index Set:**

- $I$ : Set of all products in Salesorders.csv whose 'Product Name' contains '27in'.

**Parameters:**

- $A_i$ : Revenue per unit of product $i \in I$ (from column 'Revenue').
- $d_i$ : Demand for product $i \in I$ (from column 'Demand').
- $I_i$ : Initial inventory for product $i \in I$ (from column 'Initial Inventory').

**Decision Variables:**

- $x_i$ : Number of units of product $i \in I$ to fulfill, integer, $x_i \geq 0$.

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
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   $$

---

#### Data Mapping

- **Table:** Salesorders.csv (table_id: file_0_view_0)
- **Index Set $I$:** All rows where 'Product Name' contains '27in'
- **Parameter $A_i$:** Column 'Revenue'
- **Parameter $d_i$:** Column 'Demand'
- **Parameter $I_i$:** Column 'Initial Inventory'