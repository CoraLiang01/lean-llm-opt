ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $M$: set of radio models (from HiFi1 to HiFi101; see Data Mapping)
- $W$: set of workstations (from 1 to 3; see Data Mapping)

Parameters:
- $t_{w,m}$: processing time (in minutes) required at workstation $w$ per unit of model $m$ (from file_0_view_0, column: HiFiX_Minutes, row: Workstation $w$)
- $C$: total available time per workstation per day (1,440 minutes)
- $d_w$: maintenance percentage at workstation $w$ (from file_0_view_0, column: Maintenance_Percent, row: Workstation $w$)
- $C^{\text{eff}}_w = C \cdot (1 - d_w/100)$: effective daily capacity at workstation $w$

Decision Variables:
- $x_m \in \mathbb{Z}_{\geq 0}$: number of units of model $m$ to produce per day

Auxiliary Variables:
- $I_w \geq 0$: idle time at workstation $w$

Objective:
$$
\min \sum_{w \in W} I_w
$$

Constraints:
1. Idle time definition for each workstation:
$$
I_w = C^{\text{eff}}_w - \sum_{m \in M} t_{w,m} x_m, \quad \forall w \in W
$$

2. Nonnegativity and integrality:
$$
x_m \in \mathbb{Z}_{\geq 0}, \quad \forall m \in M
$$
$$
I_w \geq 0, \quad \forall w \in W
$$

3. Production cannot exceed effective capacity:
$$
\sum_{m \in M} t_{w,m} x_m \leq C^{\text{eff}}_w, \quad \forall w \in W
$$

Data Mapping:
- $M$: radio models, columns HiFi1_Minutes, ..., HiFi101_Minutes in file_0_view_0
- $W$: workstations, file_0_view_0, column Workstation
- $t_{w,m}$: file_0_view_0, row Workstation $w$, column HiFiX_Minutes for model $m$
- $d_w$: file_0_view_0, row Workstation $w$, column Maintenance_Percent
- $C$: 1,440 (from user description)
- $C^{\text{eff}}_w$: computed as above for each $w$ using $d_w$
- $x_m$: decision variable, number of units of model $m$ to produce per day
- $I_w$: auxiliary variable, idle time at workstation $w$ per day

All parameters and index sets are defined by the current records in file_0_view_0 (workstation_times.csv).