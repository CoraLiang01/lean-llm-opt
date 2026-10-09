#### Index Sets
- $M$: set of radio models (from HiFi-1 to HiFi-101)
- $W$: set of workstations ($W = \{1,2,3\}$)

#### Parameters
- $p_{w,m}$: processing time (in minutes) required at workstation $w \in W$ for one unit of model $m \in M$  
  [Data Mapping: workstation_times.csv, columns HiFi1_Minutes, ..., HiFi101_Minutes, indexed by Workstation]
- $T$: total available time per workstation per day ($T = 1440$ minutes)
- $q_w$: maintenance percentage at workstation $w$  
  [Data Mapping: workstation_times.csv, column Maintenance_Percent, indexed by Workstation]
- $C_w$: effective daily capacity at workstation $w$, $C_w = T \cdot (1 - q_w/100)$

#### Decision Variables
- $x_m \in \mathbb{Z}_+, \quad \forall m \in M$: number of units of model $m$ to produce per day
- $idle_w \geq 0, \quad \forall w \in W$: idle time (in minutes) at workstation $w$ per day

#### Objective
Minimize total idle production time across all workstations:
$$
\min \sum_{w \in W} idle_w
$$

#### Constraints

1. **Idle time definition for each workstation:**
   $$
   idle_w = C_w - \sum_{m \in M} p_{w,m} x_m, \quad \forall w \in W
   $$
2. **Nonnegativity of idle time:**
   $$
   idle_w \geq 0, \quad \forall w \in W
   $$
3. **Nonnegativity and integrality of production quantities:**
   $$
   x_m \in \mathbb{Z}_+, \quad \forall m \in M
   $$

#### Data Mapping

- Table: workstation_times.csv (table_id: file_0_view_0)
    - Index set $W$ (workstations): column "Workstation"
    - Index set $M$ (models): columns "HiFi1_Minutes" through "HiFi101_Minutes"
    - Parameter $p_{w,m}$: value in column "HiFiX_Minutes" for model $m$ and row with "Workstation" $w$
    - Parameter $q_w$: column "Maintenance_Percent" for each workstation row
    - All 3 workstation rows and all 101 model columns are used as returned by CSVQA

No additional constraints or data sources are used. All parameters and sets are defined exactly as mapped from the returned data.