Mathematical Model

Index Sets:
- $M$: set of radio models, as given by all columns of the form HiFi$k$_Minutes in table_id file_0_view_0, $k=1,\ldots,101$
- $W$: set of workstations, as given by the Workstation column in table_id file_0_view_0

Parameters:
- $t_{wm}$: processing time (in minutes) required per unit of model $m \in M$ at workstation $w \in W$; from column $m$ in row with Workstation $w$ in file_0_view_0
- $C$: total available time per workstation per day (1,440 minutes)
- $d_w$: maintenance percentage at workstation $w$; from Maintenance_Percent column in row with Workstation $w$ in file_0_view_0
- $c_w = C \cdot (1 - d_w/100)$: effective daily capacity (in minutes) at workstation $w$

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
I_w = c_w - \sum_{m \in M} t_{wm} x_m, \quad \forall w \in W
$$

2. Nonnegativity and integrality:
$$
x_m \in \mathbb{Z}_{\geq 0}, \quad \forall m \in M
$$
$$
I_w \geq 0, \quad \forall w \in W
$$

Data Mapping

- $M$: All columns in file_0_view_0 with names matching HiFi$k$_Minutes, $k=1,\ldots,101$
- $W$: All values in Workstation column of file_0_view_0
- $t_{wm}$: For $w \in W$, $m \in M$, value in column $m$ and row with Workstation $w$ in file_0_view_0
- $C$: 1,440 (fixed)
- $d_w$: Maintenance_Percent column in row with Workstation $w$ in file_0_view_0
- $c_w$: $C \cdot (1 - d_w/100)$
- $x_m$: decision variable, nonnegative integer, for each $m \in M$
- $I_w$: auxiliary variable, nonnegative continuous, for each $w \in W$