#### Abstract Mathematical Model

**Index Sets:**
- $M$: set of radio models (from HiFi-1 to HiFi-101; see Data Mapping)
- $W$: set of workstations (from workstation_times.csv, column Workstation)

**Parameters:**
- $t_{w,m}$: processing time (in minutes) required per unit of model $m$ at workstation $w$ (from workstation_times.csv, columns HiFiX_Minutes, indexed by Workstation)
- $C_w$: total available minutes per day at workstation $w$ (given: $C_w = 1440$ for all $w$)
- $p_w$: maintenance percent at workstation $w$ (from workstation_times.csv, column Maintenance_Percent)
- $E_w$: effective daily capacity at workstation $w$, $E_w = C_w \cdot (1 - p_w/100)$

**Decision Variables:**
- $x_m \in \mathbb{Z}_{\geq 0}$: number of units of model $m$ to produce per day

**Auxiliary Variables:**
- $I_w \geq 0$: idle production time (in minutes) at workstation $w$

---

**Objective:**
\[
\min \sum_{w \in W} I_w
\]

**Constraints:**
1. **Idle time definition for each workstation:**
   \[
   I_w = E_w - \sum_{m \in M} t_{w,m} x_m, \quad \forall w \in W
   \]
2. **Nonnegativity of idle time:**
   \[
   I_w \geq 0, \quad \forall w \in W
   \]
3. **Nonnegativity and integrality of production:**
   \[
   x_m \in \mathbb{Z}_{\geq 0}, \quad \forall m \in M
   \]

---

#### Data Mapping

- $M$ (radio models): columns HiFi1_Minutes, HiFi2_Minutes, ..., HiFi101_Minutes in file_0_view_0 (workstation_times.csv)
- $W$ (workstations): column Workstation in file_0_view_0 (workstation_times.csv)
- $t_{w,m}$: value in column HiFiX_Minutes for row with Workstation $w$ in file_0_view_0
- $p_w$: value in column Maintenance_Percent for row with Workstation $w$ in file_0_view_0
- $C_w$: constant 1440 for all $w$
- $E_w$: $E_w = 1440 \cdot (1 - p_w/100)$ for each $w$ (using Maintenance_Percent from file_0_view_0)

All parameters and index sets are mapped directly from file_0_view_0 (workstation_times.csv), preserving all original columns and row order.