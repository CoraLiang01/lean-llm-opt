#### Abstract Mathematical Model

**Index Sets:**
- $C$: set of components (from processing_time_unit.csv columns C1–C111, unit_price.csv Unnamed: 0)
- $W$: set of workshops (from processing_time_unit.csv Unnamed: 0, total_working_hours.csv workshop)

**Parameters:**
- $t_{w,c}$: unit processing time of component $c$ in workshop $w$ (from processing_time_unit.csv, row $w$, column $c$)
- $p_c$: unit price of component $c$ (from unit_price.csv, row $c$, column unit_price)
- $T_w$: total available working hours in workshop $w$ (from total_working_hours.csv, row $w$, column total_hours)

**Decision Variables:**
- $x_c \in \mathbb{Z}_{\geq 0}$: number of units to produce of component $c$

**Objective:**
\[
\max \sum_{c \in C} p_c \, x_c
\]

**Constraints:**
\[
\sum_{c \in C} t_{w,c} \, x_c \leq T_w \qquad \forall w \in W
\]
\[
x_c \in \mathbb{Z}_{\geq 0} \qquad \forall c \in C
\]

---

#### Data Mapping

- $C$: processing_time_unit.csv columns C1–C111; unit_price.csv Unnamed: 0
- $W$: processing_time_unit.csv Unnamed: 0; total_working_hours.csv workshop
- $t_{w,c}$: processing_time_unit.csv, row [Unnamed: 0 = $w$], column $c$
- $p_c$: unit_price.csv, row [Unnamed: 0 = $c$], column unit_price
- $T_w$: total_working_hours.csv, row [workshop = $w$], column total_hours

- $x_c$: number of units to produce of component $c$ (decision variable, nonnegative integer)