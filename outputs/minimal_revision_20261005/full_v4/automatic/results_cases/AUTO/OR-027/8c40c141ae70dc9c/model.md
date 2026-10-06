#### Index Sets

- Let $\mathcal{I}$ be the set of all products classified under ‘Organ’ in the dataset.

#### Parameters

- $A_i$: Revenue per unit of product $i \in \mathcal{I}$ (from column "Revenue")
- $d_i$: Total deterministic demand for product $i \in \mathcal{I}$ (from column "Demand")
- $I_i$: Initial inventory for product $i \in \mathcal{I}$ (from column "Initial Inventory")

#### Decision Variables

- $x_i$: Number of units of product $i \in \mathcal{I}$ to fulfill, $x_i \in \mathbb{Z}_+$

#### Objective

$$
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
$$

#### Constraints

1. **Inventory Constraint**  
   $$
   x_i \leq I_i \qquad \forall i \in \mathcal{I}
   $$

2. **Demand Constraint**  
   $$
   x_i \leq d_i \qquad \forall i \in \mathcal{I}
   $$

3. **Non-negativity and Integrality**  
   $$
   x_i \in \mathbb{Z}_+, \qquad \forall i \in \mathcal{I}
   $$

---

#### Data Mapping

- **Index Set $\mathcal{I}$**: All products classified under ‘Organ’ in  
  `SupermartGrocerySales-RetailAnalyticsDataset.csv`, column `"Sub Category"`
- **Parameter $A_i$**: `SupermartGrocerySales-RetailAnalyticsDataset.csv`, column `"Revenue"`
- **Parameter $d_i$**: `SupermartGrocerySales-RetailAnalyticsDataset.csv`, column `"Demand"`
- **Parameter $I_i$**: `SupermartGrocerySales-RetailAnalyticsDataset.csv`, column `"Initial Inventory"`