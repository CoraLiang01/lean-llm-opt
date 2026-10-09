##### Mathematical Model

**Index Sets:**
- $M$: set of radio models (from HiFi-1 to HiFi-101)
- $W$: set of workstations (from 1 to 3)

**Parameters:**
- $t_{wm}$: processing time (in minutes) required for one unit of model $m \in M$ at workstation $w \in W$
- $C_w$: total available production time per day at workstation $w \in W$ (1,440 minutes)
- $p_w$: maintenance percentage at workstation $w \in W$ (from Maintenance_Percent column)
- $E_w$: effective daily production time at workstation $w \in W$, $E_w = C_w \cdot (1 - p_w/100)$

**Decision Variables:**
- $x_m \in \mathbb{Z}_{\geq 0}$: number of units of model $m \in M$ to produce per day

**Auxiliary Variables:**
- $I_w \geq 0$: idle time (in minutes) at workstation $w \in W$

**Objective:**
\[
\min \sum_{w \in W} I_w
\]

**Constraints:**
1. **Idle time definition for each workstation:**
   \[
   I_w = E_w - \sum_{m \in M} t_{wm} x_m, \quad \forall w \in W
   \]
2. **Nonnegativity of idle time:**
   \[
   I_w \geq 0, \quad \forall w \in W
   \]
3. **Nonnegativity and integrality of production quantities:**
   \[
   x_m \in \mathbb{Z}_{\geq 0}, \quad \forall m \in M
   \]

##### Data Mapping

- $M$: All radio model columns in `file_0_view_0` with names matching `HiFi*_Minutes`
- $W$: All rows in `file_0_view_0`, indexed by `Workstation`
- $t_{wm}$: Value in column `<model>_Minutes` for row with `Workstation = w` in `file_0_view_0`
- $C_w$: Constant $1,440$ for all $w$
- $p_w$: Value in column `Maintenance_Percent` for row with `Workstation = w` in `file_0_view_0`
- $E_w$: $C_w \cdot (1 - p_w/100)$ for each $w$
- $x_m$: Decision variable for each $m \in M$
- $I_w$: Auxiliary variable for each $w \in W$