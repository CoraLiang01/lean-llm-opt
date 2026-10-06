#### Sets and Indices

- Let \( M = \{1, 2, \ldots, 101\} \) index the radio models (HiFi-1, HiFi-2, ..., HiFi-101).
- Let \( W = \{1, 2, 3\} \) index the workstations.

#### Parameters

From the CSV (preserving file and row order):

- Let \( t_{w,m} \) = processing time (in minutes) required at workstation \( w \) for one unit of model \( m \):

  - For workstation 1: \( t_{1,m} \) = value in row 1, column "HiFi\(m\)_Minutes"
  - For workstation 2: \( t_{2,m} \) = value in row 2, column "HiFi\(m\)_Minutes"
  - For workstation 3: \( t_{3,m} \) = value in row 3, column "HiFi\(m\)_Minutes"

- Total available time per workstation per day: \( T = 1440 \) minutes.

- Maintenance percentages (from the CSV, column "Maintenance_Percent"):
  - Workstation 1: \( \mu_1 = 10\% \)
  - Workstation 2: \( \mu_2 = 14\% \)
  - Workstation 3: \( \mu_3 = 12\% \)

- Effective daily capacity at workstation \( w \):
  \[
  C_w = T \times (1 - \mu_w/100)
  \]
  That is:
  - \( C_1 = 1440 \times 0.90 = 1296 \) minutes
  - \( C_2 = 1440 \times 0.86 = 1238.4 \) minutes
  - \( C_3 = 1440 \times 0.88 = 1267.2 \) minutes

#### Decision Variables

- \( x_m \in \mathbb{Z}_{\geq 0} \): Number of units of radio model \( m \) to produce per day.

#### Auxiliary Variables

- \( \text{Idle}_w \geq 0 \): Idle time (in minutes) at workstation \( w \).

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
   (Idle time is effective capacity minus total processing time used.)

2. **Nonnegativity of idle time:**
   \[
   \text{Idle}_w \geq 0, \quad \forall w \in W
   \]

3. **Production cannot exceed effective capacity:**
   \[
   \sum_{m \in M} t_{w,m} x_m \leq C_w, \quad \forall w \in W
   \]
   (This is implied by the idle time definition and nonnegativity, but can be included explicitly.)

4. **Nonnegativity and integrality of production:**
   \[
   x_m \in \mathbb{Z}_{\geq 0}, \quad \forall m \in M
   \]

#### Full Mathematical Model

\[
\begin{align*}
\min \quad & \sum_{w=1}^3 \text{Idle}_w \\
\text{s.t.} \quad & \text{Idle}_w = C_w - \sum_{m=1}^{101} t_{w,m} x_m, \quad w=1,2,3 \\
& \text{Idle}_w \geq 0, \quad w=1,2,3 \\
& x_m \in \mathbb{Z}_{\geq 0}, \quad m=1,\ldots,101
\end{align*}
\]

Where:

- \( t_{w,m} \) are the processing times from the CSV, for each workstation \( w \) and model \( m \) (HiFi-1 to HiFi-101, in the original column order).
- \( C_1 = 1296 \), \( C_2 = 1238.4 \), \( C_3 = 1267.2 \) (minutes).

**All coefficients and identifiers are as retrieved from the CSV.**