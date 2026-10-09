ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $M$: set of radio models (from HiFi1 to HiFi101), indexed by $m$
- $W$: set of workstations (from 1 to 3), indexed by $w$

Parameters:
- $t_{w,m}$: processing time (in minutes) required at workstation $w$ per unit of model $m$  
  (from file_0_view_0, column: HiFiX_Minutes, row: Workstation $w$)
- $C$: total available minutes per workstation per day ($C = 1440$)
- $r_w$: maintenance percent at workstation $w$ (from file_0_view_0, column: Maintenance_Percent, row: Workstation $w$)
- $E_w$: effective daily capacity at workstation $w$, $E_w = C \cdot (1 - r_w/100)$

Variables:
- $x_m \in \mathbb{Z}_{\geq 0}$: number of units of model $m$ to produce per day
- $I_w \geq 0$: idle time (in minutes) at workstation $w$

Objective:
$\min \sum_{w \in W} I_w$

Constraints:
1. Idle time definition for each workstation:
   $$
   I_w = E_w - \sum_{m \in M} t_{w,m} x_m \quad \forall w \in W
   $$
2. Nonnegativity of idle time:
   $$
   I_w \geq 0 \quad \forall w \in W
   $$
3. Nonnegativity and integrality of production:
   $$
   x_m \in \mathbb{Z}_{\geq 0} \quad \forall m \in M
   $$

Data Mapping:
- $M$: All columns in file_0_view_0 with names matching "HiFiX_Minutes" for $X=1$ to $101$
- $W$: file_0_view_0, column "Workstation"
- $t_{w,m}$: file_0_view_0, row with Workstation $w$, column "HiFiX_Minutes" for model $m$
- $r_w$: file_0_view_0, row with Workstation $w$, column "Maintenance_Percent"
- $C$: fixed at 1440
- $E_w$: $C \cdot (1 - r_w/100)$ for each $w$ (computed from above)
- $x_m$: decision variable, one per model $m$
- $I_w$: decision variable, one per workstation $w$