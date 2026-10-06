#### Abstract Mathematical Model

**Index Sets:**
- $M$: set of radio models (from HiFi-1 to HiFi-101; see Data Mapping)
- $W$: set of workstations (from workstation_times.csv, column Workstation)

**Parameters:**
- $t_{w,m}$: processing time (in minutes) required per unit of model $m$ at workstation $w$ (from workstation_times.csv, columns HiFi1_Minutes, ..., HiFi101_Minutes, indexed by Workstation)
- $C_w$: total available time per day at workstation $w$ (given as 1,440 minutes for all $w$)
- $p_w$: maintenance percent at workstation $w$ (from workstation_times.csv, column Maintenance_Percent, indexed by Workstation)
- $E_w$: effective daily capacity at workstation $w$, $E_w = C_w \cdot (1 - p_w/100)$

**Decision Variables:**
- $x_m \in \mathbb{Z}_{\geq 0}$: number of units of model $m$ to produce per day

**Auxiliary Variables:**
- $I_w \geq 0$: idle time (in minutes) at workstation $w$

**Objective:**
$$
\min \sum_{w \in W} I_w
$$

**Constraints:**
1. **Idle time definition for each workstation:**
   $$
   I_w = E_w - \sum_{m \in M} t_{w,m} x_m, \quad \forall w \in W
   $$
2. **Nonnegativity of idle time:**
   $$
   I_w \geq 0, \quad \forall w \in W
   $$
3. **Nonnegativity and integrality of production:**
   $$
   x_m \in \mathbb{Z}_{\geq 0}, \quad \forall m \in M
   $$

---

#### Data Mapping

- $M$ (radio models): Columns HiFi1_Minutes, ..., HiFi101_Minutes in table_id: file_0_view_0
- $W$ (workstations): Column Workstation in table_id: file_0_view_0
- $t_{w,m}$: Value in column HiFiX_Minutes for model $m$ and row with Workstation $w$ in table_id: file_0_view_0
- $p_w$: Value in column Maintenance_Percent for row with Workstation $w$ in table_id: file_0_view_0
- $C_w$: 1,440 for all $w$ (given in user description)
- $E_w$: $E_w = 1,440 \cdot (1 - p_w/100)$ for each $w$ (maintenance percent from file_0_view_0)

All parameters and index sets are mapped directly from the columns and rows of workstation_times.csv as described above. No data is omitted or synthesized.