#### Mathematical Model

Let:
- $M$ = set of radio models (from "HiFi1" to "HiFi101", as columns in workstation_times.csv)
- $W$ = set of workstations (from "Workstation" column in workstation_times.csv)
- $t_{wm}$ = processing time (in minutes) required per unit of model $m \in M$ at workstation $w \in W$ (from "file_0_view_0", columns "HiFiX_Minutes")
- $p_w$ = maintenance percent at workstation $w$ (from "file_0_view_0", column "Maintenance_Percent")
- $C$ = total available minutes per workstation per day (given as 1,440)
- $x_m$ = number of units of model $m$ to produce per day (decision variable, integer, $\geq 0$)
- $I_w$ = idle time at workstation $w$ (auxiliary variable, continuous, $\geq 0$)

**Objective:**
$$
\min \sum_{w \in W} I_w
$$

**Constraints:**
1. **Workstation time usage and idle time definition:**
   $$
   \sum_{m \in M} t_{wm} x_m + I_w = (1 - p_w/100) \cdot C, \quad \forall w \in W
   $$
   (Effective capacity is total time minus maintenance.)

2. **Nonnegativity and integrality:**
   $$
   x_m \in \mathbb{Z}_{\geq 0}, \quad \forall m \in M
   $$
   $$
   I_w \geq 0, \quad \forall w \in W
   $$

---

#### Data Mapping

- $M$ (radio models): All columns in "file_0_view_0" with names matching "HiFiX_Minutes" (excluding "Workstation" and "Maintenance_Percent").
- $W$ (workstations): All values in "Workstation" column of "file_0_view_0".
- $t_{wm}$: For each $w$ in "Workstation", and $m$ in "HiFiX_Minutes" columns, use value at ("file_0_view_0", row with Workstation $w$, column $m$).
- $p_w$: For each $w$ in "Workstation", use value in "Maintenance_Percent" column of "file_0_view_0".
- $C$: Scalar, 1,440 (from user description).
- $x_m$: Decision variable, nonnegative integer, for each $m \in M$.
- $I_w$: Auxiliary variable, nonnegative continuous, for each $w \in W$.

All parameter values are to be taken directly from the corresponding columns and rows of "file_0_view_0" in workstation_times.csv.