#### Index Sets
- $I$: set of vehicle types (from products.csv, column ProductName)
- $J$: set of warehouses (from capacity.csv, column Warehouse ID)

#### Parameters
- $v_i$: value (benefit coefficient) of vehicle type $i \in I$ (from products.csv, column Value)
- $w_i$: weight (storage space required) of vehicle type $i \in I$ (from products.csv, column Weight)
- $C_j$: capacity of warehouse $j \in J$ (from capacity.csv, column Capacity)

#### Decision Variables
- $x_{ij}$: integer number of vehicles of type $i \in I$ to store in warehouse $j \in J$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

#### Objective
$$
\max \sum_{j \in J} \sum_{i \in I} v_i \cdot x_{ij}
$$

#### Constraints

1. **Warehouse Capacity Constraints**  
   For each warehouse $j \in J$:
   $$
   \sum_{i \in I} w_i \cdot x_{ij} \leq C_j
   $$

2. **Non-negativity and Integrality**
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- **Vehicle types and coefficients:**  
  Table: file_1_view_0 (products.csv)  
  Columns: ProductName (vehicle type), Value (benefit coefficient), Weight (storage space per unit)

- **Warehouses and capacities:**  
  Table: file_0_view_0 (capacity.csv)  
  Columns: Warehouse ID, Capacity

All records from both tables are included as returned by CSVQA. No additional filters or restrictions are applied.