#### Abstract Mathematical Model

**Index Sets:**
- $C$: set of components (from `unit_price.csv`, column `Unnamed: 0`)
- $W$: set of workshops (from `processing_time_unit.csv`, column `Unnamed: 0` and `total_working_hours.csv`, column `workshop`)

**Parameters:**
- $p_c$: unit price of component $c$ (from `unit_price.csv`, column `unit_price`)
- $a_{w,c}$: processing time required in workshop $w$ per unit of component $c$ (from `processing_time_unit.csv`, row $w$, column $c$)
- $b_w$: total available working hours in workshop $w$ (from `total_working_hours.csv`, column `total_hours`)

**Decision Variables:**
- $x_c \in \mathbb{Z}_{\geq 0}$: number of units of component $c$ to produce

**Objective:**
\[
\max \sum_{c \in C} p_c \, x_c
\]

**Constraints:**
\[
\sum_{c \in C} a_{w,c} \, x_c \leq b_w \qquad \forall w \in W
\]
\[
x_c \in \mathbb{Z}_{\geq 0} \qquad \forall c \in C
\]

---

#### Data Mapping

- $C$: All values in `unit_price.csv`, column `Unnamed: 0`
- $W$: All values in `processing_time_unit.csv`, column `Unnamed: 0` (identical to `total_working_hours.csv`, column `workshop`)
- $p_c$: `unit_price.csv`, column `unit_price`, keyed by `Unnamed: 0`
- $a_{w,c}$: `processing_time_unit.csv`, value at row with `Unnamed: 0 = w`, column $c$
- $b_w$: `total_working_hours.csv`, column `total_hours`, keyed by `workshop`
- $x_c$: Decision variable for each $c \in C$ (component)

All index sets and parameters are defined by the full set of records in the respective columns as described above.