## Abstract Mathematical Model

**Index Sets:**
- $M$: set of radio models (from "HiFi1" to "HiFi101"), indexed by $m$
- $W$: set of workstations (from "1" to "3"), indexed by $w$

**Parameters:**
- $t_{w,m}$: processing time (in minutes) required at workstation $w$ per unit of model $m$  
  (from column `HiFiX_Minutes` for model $m$ and row with `Workstation` $w$ in `file_0_view_0`)
- $C_w$: total available minutes per day at workstation $w$ (given: $1,440$ for all $w$)
- $p_w$: maintenance percentage at workstation $w$ (from column `Maintenance_Percent` in `file_0_view_0`)
- $E_w$: effective daily capacity at workstation $w$, $E_w = C_w \cdot (1 - p_w/100)$

**Decision Variables:**
- $x_m \in \mathbb{Z}_{\geq 0}$: number of units of model $m$ to produce per day

**Auxiliary Variables:**
- $I_w \geq 0$: idle time (in minutes) at workstation $w$

---

### Objective

$$
\min \sum_{w \in W} I_w
$$

### Constraints

1. **Idle time definition for each workstation:**
   $$
   I_w = E_w - \sum_{m \in M} t_{w,m} x_m, \quad \forall w \in W
   $$
   (with $I_w \geq 0$)

2. **Nonnegativity and integrality:**
   $$
   x_m \in \mathbb{Z}_{\geq 0}, \quad \forall m \in M
   $$
   $$
   I_w \geq 0, \quad \forall w \in W
   $$

3. **Production cannot exceed effective capacity:**
   $$
   \sum_{m \in M} t_{w,m} x_m \leq E_w, \quad \forall w \in W
   $$

---

## Data Mapping

- $t_{w,m}$:  
  From `file_0_view_0` (`workstation_times.csv`), for each row with `Workstation` $w$ and each column `HiFiX_Minutes` (where $m$ = "HiFiX"), use the value in that cell.
- $p_w$:  
  From `file_0_view_0`, column `Maintenance_Percent`, for each row with `Workstation` $w$.
- $C_w$:  
  Constant $1,440$ for all $w$ (from user description).
- $E_w$:  
  $E_w = 1,440 \cdot (1 - p_w/100)$ for each $w$.

- $x_m$:  
  Decision variable for each model $m$ ("HiFi1" to "HiFi101").

- $I_w$:  
  Auxiliary variable for each workstation $w$.

**All data is mapped using:**
- Table: `file_0_view_0` (from `workstation_times.csv`)
- Row key: `Workstation`
- Model columns: `HiFi1_Minutes`, ..., `HiFi101_Minutes`
- Maintenance: `Maintenance_Percent`

---

**Note:**  
- All 101 models and all 3 workstations are included, using the exact identifiers and columns as in the data.
- No data is omitted or synthesized. All variable domains and constraints are as specified in the user query and data.