#### Mathematical Model

Let:
- $M$ = set of radio models (from HiFi-1 to HiFi-101, as indexed in the data)
- $W$ = set of workstations (from the Workstation column)
- $t_{wm}$ = processing time (in minutes) required for one unit of model $m$ at workstation $w$ (from file_0_view_0, column "${\text{HiFi}k\_\text{Minutes}}$" for model $m$ and row with Workstation $w$)
- $C_w$ = total available minutes per day at workstation $w$ (given as 1,440)
- $p_w$ = maintenance percent at workstation $w$ (from file_0_view_0, column "Maintenance_Percent" for row with Workstation $w$)
- $E_w = C_w \cdot (1 - p_w/100)$ = effective daily capacity at workstation $w$
- $x_m$ = integer number of units of model $m$ to produce per day (decision variable, $x_m \in \mathbb{Z}_{\geq 0}$)
- $I_w$ = idle time at workstation $w$ (auxiliary variable, $I_w \geq 0$)

Objective:
\[
\min \sum_{w \in W} I_w
\]

Subject to:
\[
\sum_{m \in M} t_{wm} x_m + I_w = E_w \quad \forall w \in W
\]
\[
x_m \in \mathbb{Z}_{\geq 0} \quad \forall m \in M
\]
\[
I_w \geq 0 \quad \forall w \in W
\]

#### Data Mapping

- $M$: All radio models, indexed by the columns "HiFi1_Minutes", ..., "HiFi101_Minutes" in file_0_view_0.
- $W$: All workstations, indexed by the "Workstation" column in file_0_view_0.
- $t_{wm}$: file_0_view_0, row with Workstation $w$, column "${\text{HiFi}k\_\text{Minutes}}$" for model $m$.
- $C_w$: 1,440 for all $w$ (from user description).
- $p_w$: file_0_view_0, row with Workstation $w$, column "Maintenance_Percent".
- $E_w$: $C_w \cdot (1 - p_w/100)$.
- $x_m$: Decision variable, nonnegative integer, for each model $m$.
- $I_w$: Auxiliary variable, nonnegative continuous, for each workstation $w$.

All parameters are mapped directly from file_0_view_0 (workstation_times.csv) as described.