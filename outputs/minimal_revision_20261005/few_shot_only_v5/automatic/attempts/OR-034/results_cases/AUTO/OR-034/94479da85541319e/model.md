#### Abstract Mathematical Model

**Index Set:**  
Let $I$ be the set of all baked goods in the bakery, as identified by the "Product Name" column.

**Parameters:**  
For each $i \in I$:
- $A_i$: Revenue per unit of baked good $i$ ("Revenue" column)
- $d_i$: Total demand for baked good $i$ ("Demand" column)
- $I_i$: Initial inventory of baked good $i$ ("Initial Inventory" column)

**Decision Variables:**  
For each $i \in I$:
- $x_i$: Quantity of baked good $i$ to fulfill (integer, $x_i \geq 0$)

**Objective:**  
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**
1. **Demand fulfillment:**  
   $$
   x_i \leq d_i \quad \forall i \in I
   $$
2. **Inventory availability:**  
   $$
   x_i \leq I_i \quad \forall i \in I
   $$
3. **Non-negativity and integrality:**  
   $$
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   $$

---

#### Data Mapping

- **Index set $I$:**  
  Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM25/Frenchbakerydailysales.csv`, column: `Product Name`
- **Parameter $A_i$:**  
  Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM25/Frenchbakerydailysales.csv`, column: `Revenue`
- **Parameter $d_i$:**  
  Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM25/Frenchbakerydailysales.csv`, column: `Demand`
- **Parameter $I_i$:**  
  Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM25/Frenchbakerydailysales.csv`, column: `Initial Inventory`