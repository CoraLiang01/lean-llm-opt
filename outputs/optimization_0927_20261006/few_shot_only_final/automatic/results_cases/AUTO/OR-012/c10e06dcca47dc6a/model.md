---

### Sets

- $I$ : Set of products, indexed by $i$.

### Parameters

- $A_i$ : Revenue per unit of product $i$.  
  (from column `Revenue` in table `file_0_view_0`)
- $d_i$ : Total demand for product $i$ over the sales horizon.  
  (from column `Demand` in table `file_0_view_0`)
- $I_i$ : Initial inventory of product $i$.  
  (from column `Initial Inventory` in table `file_0_view_0`)

### Decision Variables

- $x_i$ : Number of units of product $i$ to fulfill for customer purchases.  
  ($x_i \in \mathbb{Z}_+, \forall i \in I$)

### Objective

\[
\max \quad \sum_{i \in I} A_i \cdot x_i
\]

### Constraints

1. **Inventory and Demand Fulfillment Bounds**  
   For all $i \in I$:
   \[
   0 \leq x_i \leq \min\{d_i, I_i\}
   \]

   (Equivalently, two constraints per $i$:)
   \[
   x_i \leq d_i
   \]
   \[
   x_i \leq I_i
   \]
   \[
   x_i \geq 0
   \]
   \[
   x_i \in \mathbb{Z}
   \]

---

### Data Mapping

- Table: `file_0_view_0` (from `OnlineSalesDataset.csv`)
    - Product identifier: `Product Name`
    - Revenue per unit: `Revenue`
    - Demand: `Demand`
    - Initial inventory: `Initial Inventory`

---