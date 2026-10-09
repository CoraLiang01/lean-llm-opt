#### Index Sets

- $W$: set of warehouses (from capacity.csv, column "Warehouse ID")
- $P$: set of vehicle types (from products.csv, column "ProductName")

#### Parameters

- $C_w$: capacity of warehouse $w \in W$ (from capacity.csv, column "Capacity")
- $v_p$: value (benefit coefficient) of vehicle type $p \in P$ (from products.csv, column "Value")
- $a_p$: weight (space requirement) of vehicle type $p \in P$ (from products.csv, column "Weight")

#### Decision Variables

- $x_{w,p} \in \mathbb{Z}_{\geq 0}$: number of vehicles of type $p$ to store in warehouse $w$ per day

#### Objective

$$
\max \sum_{w \in W} \sum_{p \in P} v_p \cdot x_{w,p}
$$

#### Constraints

1. **Warehouse Capacity Constraints**  
   For each warehouse $w \in W$:
   $$
   \sum_{p \in P} a_p \cdot x_{w,p} \leq C_w
   $$

2. **Non-negativity and Integrality**
   $$
   x_{w,p} \in \mathbb{Z}_{\geq 0}, \quad \forall w \in W,\, p \in P
   $$

---

#### Data Mapping

- Table: capacity.csv (table_id: file_0_view_0)
  - Warehouse set $W$ from column "Warehouse ID"
  - Capacity parameter $C_w$ from column "Capacity"
- Table: products.csv (table_id: file_1_view_0)
  - Vehicle type set $P$ from column "ProductName"
  - Value parameter $v_p$ from column "Value"
  - Weight parameter $a_p$ from column "Weight"