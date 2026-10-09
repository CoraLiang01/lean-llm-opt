Mathematical Model (Abstract Formulation)

Index Sets:
- \( W \): Set of workstations, indexed by \( w \) (from the Workstation column in file_0_view_0, preserving row order)
- \( M \): Set of radio models, indexed by \( m \) (from HiFi1_Minutes, ..., HiFi101_Minutes columns in file_0_view_0, preserving column order)

Parameters:
- \( T_w \): Total available time per day at workstation \( w \) (fixed at 1,440 minutes for all \( w \))
- \( \text{Maint}_w \): Maintenance percentage at workstation \( w \) (from Maintenance_Percent in file_0_view_0)
- \( C_w = T_w \times (1 - \text{Maint}_w/100) \): Effective daily production capacity (minutes) at workstation \( w \)
- \( a_{w,m} \): Processing time (minutes) required at workstation \( w \) per unit of model \( m \) (from file_0_view_0, column for \( m \) in row for \( w \))

Decision Variables:
- \( x_m \in \mathbb{Z}_+ \): Number of units of radio model \( m \) to produce per day (nonnegative integer)
- \( \text{Idle}_w \geq 0 \): Idle production time (minutes) at workstation \( w \) (continuous, nonnegative)

Objective:
\[
\min \sum_{w \in W} \text{Idle}_w
\]

Constraints:
1. Idle time definition for each workstation:
\[
\text{Idle}_w = C_w - \sum_{m \in M} a_{w,m} x_m \quad \forall w \in W
\]
2. Idle time cannot be negative:
\[
\text{Idle}_w \geq 0 \quad \forall w \in W
\]
3. Nonnegativity and integrality of production:
\[
x_m \in \mathbb{Z}_+, \quad \forall m \in M
\]

Data Mapping

Index Sets:
- \( W \): file_0_view_0.Workstation (row order: 1, 2, 3)
- \( M \): All columns in file_0_view_0 with names HiFi1_Minutes, ..., HiFi101_Minutes (column order as in file)

Parameters:
- \( T_w \): 1,440 for all \( w \)
- \( \text{Maint}_w \): file_0_view_0.Maintenance_Percent for each \( w \)
- \( C_w \): \( 1,440 \times (1 - \text{Maint}_w/100) \) for each \( w \)
- \( a_{w,m} \): file_0_view_0.<HiFiX_Minutes> for each \( w \) and \( m \) (row for \( w \), column for \( m \))

Variables:
- \( x_m \): Number of units of model \( m \) to produce per day (integer, nonnegative)
- \( \text{Idle}_w \): Idle time at workstation \( w \) (continuous, nonnegative)

Objective:
- Minimize total idle time: sum over \( w \) of \( \text{Idle}_w \)

Constraints:
- Idle time definition and nonnegativity as above, using mapped data.

No additional constraints or data sources are used.