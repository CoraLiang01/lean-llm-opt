### Abstract Mathematical Model

#### Index Sets
- $C$: set of components (from processing_time_unit.csv column headers "C1", ..., "C111")
- $W$: set of workshops (from processing_time_unit.csv column "Unnamed: 0" and total_working_hours.csv column "workshop")

#### Parameters
- $p_c$: unit price of component $c \in C$  
  (Data: unit_price.csv, column "unit_price", indexed by "Unnamed: 0" = $c$)
- $a_{w,c}$: unit processing time of component $c$ in workshop $w$  
  (Data: processing_time_unit.csv, row "Unnamed: 0" = $w$, column $c$)
- $T_w$: total available working hours in workshop $w$  
  (Data: total_working_hours.csv, column "total_hours", indexed by "workshop" = $w$)

#### Decision Variables
- $x_c$: number of units to produce of component $c \in C$  
  Domain: $x_c \geq 0$, integer

#### Objective
$$
\max \sum_{c \in C} p_c \, x_c
$$

#### Constraints

1. **Workshop Capacity Constraints**  
 For each workshop $w \in W$:
$$
\sum_{c \in C} a_{w,c} \, x_c \leq T_w
$$

2. **Non-negativity and Integrality**
$$
x_c \in \mathbb{Z}_+, \quad \forall c \in C
$$

---

#### Data Mapping

- **processing_time_unit.csv**  
 - Table ID: file_0_view_0  
 - Columns: "Unnamed: 0" (workshop, $w$), "C1"–"C111" (component, $c$)  
 - Parameter: $a_{w,c}$

- **unit_price.csv**  
 - Table ID: file_1_view_0  
 - Columns: "Unnamed: 0" (component, $c$), "unit_price"  
 - Parameter: $p_c$

- **total_working_hours.csv**  
 - Table ID: file_2_view_0  
 - Columns: "workshop" ($w$), "total_hours"  
 - Parameter: $T_w$