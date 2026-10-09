**Abstract Mathematical Model**

**Index Sets:**  
- $I$: Set of all products, indexed by $i$.

**Parameters:**  
- $A_i$: Revenue per unit for product $i$ (from column "Revenue" in table file_0_view_0).  
- $d_i$: Demand for product $i$ (from column "Demand" in table file_0_view_0).  
- $I_i$: Initial inventory for product $i$ (from column "Initial Inventory" in table file_0_view_0).

**Decision Variables:**  
- $x_i \in \mathbb{Z}_+$: Number of orders fulfilled for product $i$.

**Objective:**  
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**  
1. **Inventory and Demand Fulfillment Bounds:**  
   $$
   0 \leq x_i \leq \min\{d_i, I_i\}, \quad \forall i \in I
   $$

2. **Integrality:**  
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

**Data Mapping:**  
- Table: `file_0_view_0` (source: `MobileSalesDataset.csv`)
- Index set $I$: All records in column "Product Name".
- Parameter $A_i$: Column "Revenue".
- Parameter $d_i$: Column "Demand".
- Parameter $I_i$: Column "Initial Inventory".
- All records are included; no filters or subsets applied.  
- Each $i \in I$ corresponds to a unique row in the table.

---

**End of Model**