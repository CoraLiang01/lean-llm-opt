#### Abstract Mathematical Optimization Model

**Index Set:**

- $I$ : Set of all “4U” products (indexed by $i$).

**Parameters:**

- $A_i$ : Revenue per unit of product $i$ (from column “Revenue”).
- $d_i$ : Total demand for product $i$ over the sales horizon (from column “Demand”).
- $I_i$ : Initial inventory of product $i$ (from column “Initial Inventory”).

**Decision Variables:**

- $x_i$ : Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Inventory Constraint:** 
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$
2. **Demand Constraint:** 
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
3. **Non-negativity and Integrality:** 
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- Table: OnlineSalesinUSA.csv
- Index Set $I$: All rows where “Product Name” begins with “4U” (column “Product Name”).
- Parameter $A_i$: Column “Revenue” in OnlineSalesinUSA.csv.
- Parameter $d_i$: Column “Demand” in OnlineSalesinUSA.csv.
- Parameter $I_i$: Column “Initial Inventory” in OnlineSalesinUSA.csv.