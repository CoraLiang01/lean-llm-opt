#### Abstract Mathematical Model

**Index Sets:**

- $I$ : Set of all products classified under ‘id999’.

**Parameters:**

- $a_i$ : Revenue per unit of product $i \in I$ (from column ‘Revenue’).
- $d_i$ : Deterministic demand for product $i \in I$ during the sales horizon (from column ‘Demand’).
- $s_i$ : Initial inventory of product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**

- $x_i$ : Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integer).

**Objective:**

$$
\max \sum_{i \in I} a_i x_i
$$

**Constraints:**

1. **Inventory Constraints:**
   $$
   x_i \leq s_i \qquad \forall i \in I
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

- **Table:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv`
- **Index Set:** $I$ = all rows with `id_number = 'id999'`
- **Parameters:**
  - $a_i$ : column `Revenue`
  - $d_i$ : column `Demand`
  - $s_i$ : column `Initial Inventory`