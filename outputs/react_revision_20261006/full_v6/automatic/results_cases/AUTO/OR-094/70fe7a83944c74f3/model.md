Mathematical Model

Index Sets:
- $M$: set of radio models (from HiFi1 to HiFi101; one for each column HiFiX_Minutes in workstation_times.csv)
- $W$: set of workstations (from the Workstation column in workstation_times.csv)

Parameters:
- $t_{wm}$: processing time (in minutes) required at workstation $w \in W$ for one unit of model $m \in M$ (from column $m$ in row $w$ of workstation_times.csv)
- $C_w$: total available time per day at workstation $w \in W$ (given as 1,440 minutes for all $w$)
- $p_w$: maintenance percentage at workstation $w \in W$ (from Maintenance_Percent column in workstation_times.csv)
- $E_w = C_w \cdot (1 - p_w/100)$: effective daily capacity at workstation $w \in W$

Decision Variables:
- $x_m \in \mathbb{Z}_{\geq 0}$: number of units of model $m \in M$ to produce per day

Auxiliary Variables:
- $I_w \geq 0$: idle time (in minutes) at workstation $w \in W$

Objective:
$$
\min \sum_{w \in W} I_w
$$

Constraints:
1. Idle time definition for each workstation:
$$
I_w = E_w - \sum_{m \in M} t_{wm} x_m, \quad \forall w \in W
$$

2. Nonnegativity and integrality:
$$
x_m \in \mathbb{Z}_{\geq 0}, \quad \forall m \in M
$$
$$
I_w \geq 0, \quad \forall w \in W
$$

3. No workstation can be overloaded:
$$
\sum_{m \in M} t_{wm} x_m \leq E_w, \quad \forall w \in W
$$

Data Mapping

- $W$: file_0_view_0.Workstation
- $M$: all columns in file_0_view_0 with names matching "HiFi\d+_Minutes"
- $t_{wm}$: file_0_view_0, column $m$, row with Workstation $w$
- $C_w$: fixed at 1,440 for all $w$
- $p_w$: file_0_view_0.Maintenance_Percent, row with Workstation $w$
- $E_w$: $C_w \cdot (1 - p_w/100)$
- $x_m$: integer variable for each $m \in M$
- $I_w$: continuous variable for each $w \in W$