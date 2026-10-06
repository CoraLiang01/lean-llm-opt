#### Abstract Mathematical Model

**Index Set:**

- $I$ : Set of all products classified under ‘27in’, indexed by $i$.

**Parameters:**

- $A_i$ : Revenue per unit of product $i$.  
  [CSVQA_DATA: file_0_view_0, column: Revenue, index: Product Name]
- $I_i$ : Initial inventory of product $i$.  
  [CSVQA_DATA: file_0_view_0, column: Initial Inventory, index: Product Name]
- $d_i$ : Demand for product $i$.  
  [CSVQA_DATA: file_0_view_0, column: Demand, index: Product Name]

**Decision Variables:**

- $x_i$ : Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

**Objective Function:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Inventory Constraint:**  
  $x_i \leq I_i \quad \forall i \in I$

2. **Demand Constraint:**  
  $x_i \leq d_i \quad \forall i \in I$

3. **Non-negativity and Integrality:**  
  $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

#### Data Mapping

- $I$ : All records in [file_0_view_0] where [Product Name] has prefix ‘27in’.
- $A_i$ : [file_0_view_0], column [Revenue], indexed by [Product Name].
- $I_i$ : [file_0_view_0], column [Initial Inventory], indexed by [Product Name].
- $d_i$ : [file_0_view_0], column [Demand], indexed by [Product Name].
- $x_i$ : Decision variable for each $i \in I$.

(CSVQA_DATA bindings: file_0_view_0, columns: Product Name, Revenue, Initial Inventory, Demand; filter: Product Name prefix ‘27in’)