#### Symbolic Model

**Index Sets:**
- $M$: set of radio models (from columns HiFi1_Minutes, ..., HiFi101_Minutes in workstation_times.csv)
- $W$: set of workstations (from column Workstation in workstation_times.csv; $|W|=3$)

**Parameters:**
- $t_{wm}$: processing time (in minutes) required at workstation $w \in W$ for one unit of model $m \in M$ (from workstation_times.csv, column $m$ in row $w$)
- $C_w$: total available minutes per day at workstation $w$ (given: $C_w = 1440$ for all $w$)
- $d_w$: maintenance percentage at workstation $w$ (from workstation_times.csv, column Maintenance_Percent in row $w$)
- $E_w$: effective daily capacity at workstation $w$, $E_w = C_w \cdot (1 - d_w/100)$

**Decision Variables:**
- $x_m \in \mathbb{Z}_{\geq 0}$: number of units of model $m \in M$ to produce per day
- $s_w \geq 0$: idle time (in minutes) at workstation $w \in W$

**Objective:**
\[
\min \sum_{w \in W} s_w
\]

**Constraints:**
1. **Workstation time usage and idle time:**
   \[
   \sum_{m \in M} t_{wm} x_m + s_w = E_w, \quad \forall w \in W
   \]
2. **Nonnegativity and integrality:**
   \[
   x_m \in \mathbb{Z}_{\geq 0}, \quad \forall m \in M
   \]
   \[
   s_w \geq 0, \quad \forall w \in W
   \]

---

#### Data Mapping

- $W$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture13/workstation_times.csv, column Workstation
- $M$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture13/workstation_times.csv, columns HiFi1_Minutes through HiFi101_Minutes
- $t_{wm}$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture13/workstation_times.csv, value in column $m$ (HiFiX_Minutes) for row $w$ (Workstation)
- $d_w$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture13/workstation_times.csv, column Maintenance_Percent for row $w$ (Workstation)
- $C_w$: fixed at 1440 for all $w$
- $E_w$: computed as $C_w \cdot (1 - d_w/100)$ for each $w$ using $d_w$ above

All index sets, parameters, and coefficients are mapped directly from the supplied file and columns as specified.