**Abstract Model**

**Index Sets:**  
- $I$: Set of all products classified under "Baby".

**Parameters:**  
- $A_i$: Revenue per unit of product $i \in I$ (from column "Revenue").
- $d_i$: Demand for product $i \in I$ (from column "Demand").
- $I_i$: Initial inventory for product $i \in I$ (from column "Initial Inventory").

**Decision Variables:**  
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

**Objective:**  
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**  
1. **Inventory and Demand Fulfillment:**  
   \[
   0 \leq x_i \leq \min\{I_i, d_i\} \quad \forall i \in I
   \]

**Data Mapping:**  
- **Source Table:** `file_0_view_0` (from `Salesdata.csv`)
- **Columns Used:**  
  - "Product Name" (to select products classified under "Baby")
  - "Revenue" (parameter $A_i$)
  - "Demand" (parameter $d_i$)
  - "Initial Inventory" (parameter $I_i$)
- **Selection Criterion:**  
  - $I = \{i: \text{"Product Name"}$ contains "Baby"$\}$

**Notes:**  
- The model is abstract and applies to any set of "Baby" products identified in the source table.
- All parameters are to be taken directly from the specified columns for the selected products.