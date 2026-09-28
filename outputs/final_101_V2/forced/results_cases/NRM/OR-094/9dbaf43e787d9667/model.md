### Abstract Mathematical Model

#### Index Sets
- $M$: set of radio models (from HiFi-1 to HiFi-101)
- $W$: set of workstations ($W = \{1,2,3\}$)

#### Parameters
- $p_{w,m}$: processing time (in minutes) required per unit of model $m \in M$ at workstation $w \in W$  
  (Data: table_id = file_0_view_0, columns = "Workstation", "HiFi1_Minutes", ..., "HiFi101_Minutes")
- $T$: total available time per workstation per day ($T = 1440$ minutes)
- $r_w$: maintenance percentage at workstation $w$  
  (Data: table_id = file_0_view_0, column = "Maintenance_Percent")
- $C_w$: effective daily capacity at workstation $w$, $C_w = T \cdot (1 - r_w/100)$

#### Decision Variables
- $x_m \in \mathbb{Z}_+, \quad \forall m \in M$: number of units of model $m$ to produce per day

#### Auxiliary Variables
- $I_w \geq 0, \quad \forall w \in W$: idle production time (in minutes) at workstation $w$

#### Objective
Minimize total idle production time across all workstations:
$$
\min \sum_{w \in W} I_w
$$

#### Constraints

1. **Idle Time Definition (per workstation):**
   $$
   I_w = C_w - \sum_{m \in M} p_{w,m} x_m, \quad \forall w \in W
   $$
2. **Nonnegativity of Idle Time:**
   $$
   I_w \geq 0, \quad \forall w \in W
   $$
3. **Nonnegativity and Integrality of Production:**
   $$
   x_m \in \mathbb{Z}_+, \quad \forall m \in M
   $$

---

#### Data Mapping

- All processing times $p_{w,m}$ and maintenance percentages $r_w$ are from:
  - **table_id:** file_0_view_0
  - **columns:** "Workstation", "HiFi1_Minutes", ..., "HiFi101_Minutes", "Maintenance_Percent"
- Index sets $M$ and $W$ are determined by the columns and rows of the same table.

---

This model minimizes the total idle time across all workstations, subject to effective capacity and nonnegativity/integrality of production decisions, using all data and constraints as described in the user query and the retrieved table.