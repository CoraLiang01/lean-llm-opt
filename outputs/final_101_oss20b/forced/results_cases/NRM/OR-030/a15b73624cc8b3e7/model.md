##### Objective Function:

$\quad \quad \max \quad \sum_{i \in I} w_i \cdot x_i$

##### Constraints

###### 1. Inventory Constraints:

$\quad \quad x_i \leq I_i, \quad \forall i \in I$

###### 2. Demand Constraints:

$x_i \leq d_i, \quad \forall i \in I$

###### 3. Shared Dispatch Capacity:

$\sum_{i \in I} x_i \leq 100$

###### 4. Variable Constraints:

$x_i \in \mathbb{Z}, \quad x_i \geq 1, \quad \forall i \in I$

##### Data Mapping

{
  "table_id": "file_0_view_0",
  "columns": {
    "Revenue": "w_i",
    "Initial Inventory": "I_i",
    "Demand": "d_i"
  }
}