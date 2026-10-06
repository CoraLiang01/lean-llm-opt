#### Sets and Indices

- Let \( M = \{ \text{HiFi1}, \text{HiFi2}, \ldots, \text{HiFi101} \} \) be the set of radio models.
- Let \( W = \{1, 2, 3\} \) be the set of workstations.

#### Parameters

- \( t_{w,m} \): Processing time (in minutes) required at workstation \( w \) for one unit of model \( m \).
    - For each \( w \in W \), \( m \in M \), \( t_{w,m} \) is given by the corresponding column in the row for workstation \( w \) in workstation_times.csv.
- \( T = 1440 \): Total available minutes per workstation per day.
- \( \text{maint}_w \): Maintenance percentage at workstation \( w \):
    - \( \text{maint}_1 = 10\% \)
    - \( \text{maint}_2 = 14\% \)
    - \( \text{maint}_3 = 12\% \)
- \( C_w = T \times (1 - \text{maint}_w) \): Effective daily production capacity (in minutes) at workstation \( w \).

#### Decision Variables

- \( x_m \in \mathbb{Z}_{\geq 0} \): Number of units of model \( m \) to produce per day.

#### Auxiliary Variables

- \( \text{Idle}_w \geq 0 \): Idle time (in minutes) at workstation \( w \).

---

### Mathematical Model

#### Objective

Minimize total idle production time across all workstations:
\[
\min \sum_{w \in W} \text{Idle}_w
\]

#### Constraints

1. **Idle time definition for each workstation:**
   \[
   \text{Idle}_w = C_w - \sum_{m \in M} t_{w,m} x_m, \quad \forall w \in W
   \]
   (Alternatively, since \( \text{Idle}_w \geq 0 \), you may write:
   \[
   \sum_{m \in M} t_{w,m} x_m \leq C_w, \quad \forall w \in W
   \]
   and
   \[
   \text{Idle}_w \geq 0, \quad \forall w \in W
   \]
   )

2. **Nonnegativity and integrality:**
   \[
   x_m \in \mathbb{Z}_{\geq 0}, \quad \forall m \in M
   \]
   \[
   \text{Idle}_w \geq 0, \quad \forall w \in W
   \]

---

### Explicit Data Mapping

Let the models be indexed as follows (preserving CSV order):

- \( M = \{\text{HiFi1}, \text{HiFi2}, \ldots, \text{HiFi101}\} \)
- For each workstation \( w \in \{1,2,3\} \), the processing times \( t_{w,m} \) are given by the columns:
    - Row 1: Workstation 1, columns HiFi1_Minutes, ..., HiFi101_Minutes
    - Row 2: Workstation 2, columns HiFi1_Minutes, ..., HiFi101_Minutes
    - Row 3: Workstation 3, columns HiFi1_Minutes, ..., HiFi101_Minutes

Effective capacities:
- \( C_1 = 1440 \times 0.90 = 1296 \) minutes
- \( C_2 = 1440 \times 0.86 = 1238.4 \) minutes
- \( C_3 = 1440 \times 0.88 = 1267.2 \) minutes

---

### Complete Formulation

\[
\begin{align*}
\min \quad & \text{Idle}_1 + \text{Idle}_2 + \text{Idle}_3 \\
\text{s.t.} \quad
& \sum_{m \in M} t_{1,m} x_m + \text{Idle}_1 = 1296 \\
& \sum_{m \in M} t_{2,m} x_m + \text{Idle}_2 = 1238.4 \\
& \sum_{m \in M} t_{3,m} x_m + \text{Idle}_3 = 1267.2 \\
& x_m \in \mathbb{Z}_{\geq 0} \quad \forall m \in M \\
& \text{Idle}_w \geq 0 \quad \forall w \in \{1,2,3\}
\end{align*}
\]

Where:
- \( t_{w,m} \) are the processing times from the CSV, for each workstation \( w \) and model \( m \), using the exact column and row order as in the file.
- \( x_m \) is the number of units of model \( m \) to produce per day.

**All data and indices are to be used exactly as retrieved from the CSV, preserving order and identifiers.**