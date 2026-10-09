#### Mathematical Model

Let:
- $M$ = set of radio models (from HiFi-1 to HiFi-101)
- $W$ = set of workstations (from 1 to 3)
- $t_{wm}$ = processing time (in minutes) required for one unit of model $m \in M$ at workstation $w \in W$
- $C_w$ = total available minutes per day at workstation $w$ (1,440 minutes for all $w$)
- $p_w$ = maintenance percent at workstation $w$ (from data)
- $E_w = C_w \cdot (1 - p_w/100)$ = effective daily capacity at workstation $w$
- $x_m$ = number of units of model $m$ to produce per day (decision variable, integer $\geq 0$)
- $I_w$ = idle time at workstation $w$ (continuous, $\geq 0$)

**Variables:**
- $x_m \in \mathbb{Z}_{\geq 0}$, $\forall m \in M$
- $I_w \geq 0$, $\forall w \in W$

**Objective:**
\[
\min \sum_{w \in W} I_w
\]

**Constraints:**
1. Idle time definition for each workstation:
   \[
   I_w = E_w - \sum_{m \in M} t_{wm} x_m, \quad \forall w \in W
   \]
2. Nonnegativity of idle time:
   \[
   I_w \geq 0, \quad \forall w \in W
   \]
3. Nonnegativity and integrality of production:
   \[
   x_m \in \mathbb{Z}_{\geq 0}, \quad \forall m \in M
   \]

#### Data Mapping

- Index set $W$ (workstations): file_0_view_0, column "Workstation"
- Index set $M$ (radio models): file_0_view_0, columns "HiFi1_Minutes" through "HiFi101_Minutes" (strip "_Minutes" for model name)
- Parameter $t_{wm}$: file_0_view_0, row with "Workstation" = $w$, column "HiFiX_Minutes" for model $m$ = HiFi-X
- Parameter $C_w$: fixed at 1,440 for all $w$
- Parameter $p_w$: file_0_view_0, row with "Workstation" = $w$, column "Maintenance_Percent"
- Parameter $E_w$: $E_w = 1,440 \cdot (1 - p_w/100)$
- Variable $x_m$: number of units of model $m$ to produce per day
- Variable $I_w$: idle time at workstation $w$ per day

All parameters and index sets are mapped directly from file_0_view_0 (workstation_times.csv) as described.