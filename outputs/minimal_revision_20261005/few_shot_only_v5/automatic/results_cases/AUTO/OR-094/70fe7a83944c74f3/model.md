**Abstract Mathematical Model**

**Index Sets:**
- $W$: set of workstations (from column `Workstation` in workstation_times.csv)
- $M$: set of radio models (from columns `HiFi1_Minutes`, ..., `HiFi101_Minutes` in workstation_times.csv)

**Parameters:**
- $t_{w,m}$: processing time (in minutes) required for one unit of model $m$ at workstation $w$ (from column `HiFi{n}_Minutes` for each $m$ in workstation_times.csv, for each $w$)
- $C$: total available minutes per workstation per day (fixed at 1440)
- $mp_w$: maintenance percent at workstation $w$ (from column `Maintenance_Percent` in workstation_times.csv)
- $c^{\text{eff}}_w = C \cdot (1 - mp_w/100)$: effective daily capacity at workstation $w$ after maintenance

**Decision Variables:**
- $x_m \in \mathbb{Z}_{\geq 0}$: number of units of model $m$ to produce per day

- $I_w \geq 0$: idle time (in minutes) at workstation $w$

**Objective:**
\[
\min \sum_{w \in W} I_w
\]

**Constraints:**

1. **Idle time definition for each workstation:**
   \[
   I_w = c^{\text{eff}}_w - \sum_{m \in M} t_{w,m} x_m \qquad \forall w \in W
   \]
   (Alternatively, $I_w \geq c^{\text{eff}}_w - \sum_{m \in M} t_{w,m} x_m$, $I_w \geq 0$)

2. **Production cannot exceed effective capacity:**
   \[
   \sum_{m \in M} t_{w,m} x_m \leq c^{\text{eff}}_w \qquad \forall w \in W
   \]

3. **Nonnegativity and integrality:**
   \[
   x_m \in \mathbb{Z}_{\geq 0} \qquad \forall m \in M
   \]
   \[
   I_w \geq 0 \qquad \forall w \in W
   \]

---

**Data Mapping**

- $W$: workstation_times.csv, column `Workstation`
- $M$: workstation_times.csv, columns `HiFi1_Minutes` through `HiFi101_Minutes` (model $m$ corresponds to column `HiFi{n}_Minutes`)
- $t_{w,m}$: workstation_times.csv, value at row with `Workstation` = $w$, column `HiFi{n}_Minutes` for model $m$
- $mp_w$: workstation_times.csv, value at row with `Workstation` = $w$, column `Maintenance_Percent`
- $C$: fixed at 1440 (per query)
- $c^{\text{eff}}_w$: computed as $1440 \cdot (1 - mp_w/100)$ for each $w$ (using `Maintenance_Percent`)

**Variables:**
- $x_m$: number of units of model $m$ to produce per day (integer, nonnegative)
- $I_w$: idle time at workstation $w$ (continuous, nonnegative)