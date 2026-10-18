#### Index Sets

- $M$: set of radio models (HiFi-1, HiFi-2, ..., HiFi-101)
- $W$: set of workstations ($W = \{1,2,3\}$)

#### Parameters

- $p_{w,m}$: processing time (in minutes) required at workstation $w \in W$ for one unit of model $m \in M$  
  (from file_0_view_0, columns: "Workstation", "HiFi1_Minutes", ..., "HiFi101_Minutes")
- $T$: total available time per workstation per day ($T = 1440$ minutes)
- $r_w$: maintenance percentage at workstation $w$  
  (from file_0_view_0, column: "Maintenance_Percent")
- $C_w$: effective daily capacity at workstation $w$  
  $C_w = T \cdot (1 - r_w/100)$

#### Decision Variables

- $x_m \in \mathbb{Z}_+, \quad \forall m \in M$  
  Number of units of model $m$ to produce per day (nonnegative integer)

- $idle_w \geq 0, \quad \forall w \in W$  
  Idle time (in minutes) at workstation $w$

#### Objective

Minimize total idle production time across all workstations:
$$
\min \sum_{w \in W} idle_w
$$

#### Constraints

1. Idle time definition for each workstation:
   $$
   idle_w = C_w - \sum_{m \in M} p_{w,m} x_m, \quad \forall w \in W
   $$
2. Nonnegativity of idle time:
   $$
   idle_w \geq 0, \quad \forall w \in W
   $$
3. Nonnegativity and integrality of production quantities:
   $$
   x_m \in \mathbb{Z}_+, \quad \forall m \in M
   $$

#### Data Mapping

- Table: workstation_times.csv (table_id: file_0_view_0)
    - Model set $M$ and processing times $p_{w,m}$: columns "HiFi1_Minutes", ..., "HiFi101_Minutes"
    - Workstation set $W$ and maintenance percentages $r_w$: columns "Workstation", "Maintenance_Percent"
    - All 3 rows (one per workstation) are used; all 101 model columns are used.

No additional filtering or selection is applied; all data from the returned records are used as described.