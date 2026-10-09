#### Abstract Mathematical Model

**Index Sets:**
- $M$: set of radio models (from HiFi-1 to HiFi-101; indexed by $m$)
- $W$: set of workstations (from 1 to 3; indexed by $w$)

**Parameters:**
- $t_{w,m}$: processing time (in minutes) required at workstation $w$ per unit of model $m$  
  (from column `HiFi{n}_Minutes` in `file_0_view_0`, where $n$ is the model number, and row with `Workstation = w$)
- $C_w$: total available minutes per day at workstation $w$ (given as 1,440 for all $w$)
- $d_w$: maintenance percentage at workstation $w$ (from column `Maintenance_Percent` in `file_0_view_0`)
- $E_w$: effective daily capacity at workstation $w$, $E_w = C_w \cdot (1 - d_w/100)$

**Decision Variables:**
- $x_m \in \mathbb{Z}_{\geq 0}$: number of units of model $m$ to produce per day

**Auxiliary Variables:**
- $I_w \geq 0$: idle time (in minutes) at workstation $w$

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

- $M$: All radio models, corresponding to columns `HiFi1_Minutes`, ..., `HiFi101_Minutes` in `file_0_view_0`
- $W$: All workstations, corresponding to rows with `Workstation` values in `file_0_view_0`
- $t_{w,m}$: Value in column `HiFi{n}_Minutes` for row with `Workstation = w` in `file_0_view_0`
- $C_w$: 1,440 for all $w$ (from user description)
- $d_w$: Value in column `Maintenance_Percent` for row with `Workstation = w` in `file_0_view_0`
- $E_w$: $C_w \cdot (1 - d_w/100)$, computed per workstation
- $x_m$: Decision variable, number of units of model $m$ to produce per day
- $I_w$: Auxiliary variable, idle time at workstation $w$

All parameters are mapped directly from `file_0_view_0` (workstation_times.csv) as described above.