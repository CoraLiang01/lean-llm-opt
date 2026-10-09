#### Index Sets

- $I$: Set of vehicle types (from products.csv, column ProductName).

#### Parameters

- $v_i$: Profit per unit of vehicle type $i \in I$ (from products.csv, column Value).
- $w_i$: Inventory weight per unit of vehicle type $i \in I$ (from products.csv, column Weight).
- $C$: Overall inventory capacity (from capacity.csv, column Capacity).

#### Decision Variables

- $x_i$: Number of vehicles of type $i \in I$ to order per day. ($x_i \in \mathbb{Z}_{\geq 0}$)

#### Objective

$$
\max \sum_{i \in I} v_i \cdot x_i
$$

#### Constraints

1. **Overall Inventory Capacity:**
   $$
   \sum_{i \in I} w_i \cdot x_i \leq C
   $$

2. **Non-negativity and Integrality:**
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   $$

---

#### Data Mapping

- Table `file_1_view_0` (products.csv): 
  - $I$ = ProductName
  - $v_i$ = Value
  - $w_i$ = Weight
- Table `file_0_view_0` (capacity.csv): 
  - $C$ = Capacity

All records from both tables are included as returned by CSVQA.