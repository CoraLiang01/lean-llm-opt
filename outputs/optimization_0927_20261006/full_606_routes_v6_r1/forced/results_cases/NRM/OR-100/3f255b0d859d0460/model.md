#### Index Sets
- $I$: set of component types (from file_1_view_0, column "Unnamed: 0")
- $W$: set of workshops (from file_0_view_0, column "Unnamed: 0" and file_2_view_0, column "workshop")

#### Parameters
- $p_i$: unit price of component $i \in I$ (from file_1_view_0, column "unit_price")
- $t_{w,i}$: unit processing time of component $i \in I$ in workshop $w \in W$ (from file_0_view_0, row "Unnamed: 0" = $w$, column $i$)
- $T_w$: total available working hours in workshop $w \in W$ (from file_2_view_0, columns "workshop", "total_hours")

#### Decision Variables
- $x_i \in \mathbb{Z}_+, \quad \forall i \in I$: production quantity of component $i$

#### Objective
\[
\max \sum_{i \in I} p_i x_i
\]

#### Constraints
- Workshop capacity constraints:
  \[
  \sum_{i \in I} t_{w,i} x_i \leq T_w, \quad \forall w \in W
  \]
- Nonnegativity and integrality:
  \[
  x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in I
  \]

---

#### Data Mapping

- file_0_view_0 (processing_time_unit.csv): 
  - Index set $W$ from column "Unnamed: 0" (workshop names)
  - Index set $I$ from columns "C1" to "C111" (component types)
  - Parameter $t_{w,i}$ from row "Unnamed: 0" = $w$, column $i$
- file_1_view_0 (unit_price.csv): 
  - Index set $I$ from column "Unnamed: 0"
  - Parameter $p_i$ from column "unit_price"
- file_2_view_0 (total_working_hours.csv): 
  - Index set $W$ from column "workshop"
  - Parameter $T_w$ from column "total_hours"

All rows and columns from each file are used as returned by CSVQA.