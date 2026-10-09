Let \( x_j \) be the nonnegative integer number of units of radio model \( j \) (for \( j = 1, \ldots, 101 \), corresponding to HiFi1, HiFi2, ..., HiFi101) to produce per day.

Let \( t_{i,j} \) be the processing time (in minutes) required per unit of model \( j \) at workstation \( i \), as given in workstation_times.csv. The data is as follows (with columns in original order):

- Workstation 1: Maintenance_Percent = 10%
- Workstation 2: Maintenance_Percent = 14%
- Workstation 3: Maintenance_Percent = 12%

Total available time per workstation per day: 1,440 minutes.

Effective daily capacities:
- Workstation 1: \( 1,440 \times (1 - 0.10) = 1,296 \) minutes
- Workstation 2: \( 1,440 \times (1 - 0.14) = 1,238.4 \) minutes
- Workstation 3: \( 1,440 \times (1 - 0.12) = 1,267.2 \) minutes

Decision variables:
- \( x_j \in \mathbb{Z}_+ \) for \( j = 1, \ldots, 101 \)

Parameters (from workstation_times.csv, preserving order):

\[
\begin{array}{l|cccccc}
\text{Workstation} & \text{HiFi1\_Minutes} & \text{HiFi2\_Minutes} & \cdots & \text{HiFi101\_Minutes} \\
\hline
1 & 6 & 4 & \cdots & 10 \\
2 & 5 & 5 & \cdots & 14 \\
3 & 4 & 6 & \cdots & 12 \\
\end{array}
\]

Objective:
Minimize total idle production time across all three workstations:
\[
\text{Minimize} \quad \sum_{i=1}^3 \left( \text{EffectiveCapacity}_i - \sum_{j=1}^{101} t_{i,j} x_j \right )
\]
which is equivalent to:
\[
\text{Minimize} \quad \left( 1,296 - \sum_{j=1}^{101} t_{1,j} x_j \right ) + \left( 1,238.4 - \sum_{j=1}^{101} t_{2,j} x_j \right ) + \left( 1,267.2 - \sum_{j=1}^{101} t_{3,j} x_j \right )
\]
or, equivalently (since the constants do not affect the optimizer's solution, but must be preserved for the objective value):
\[
\text{Minimize} \quad 3,801.6 - \sum_{i=1}^3 \sum_{j=1}^{101} t_{i,j} x_j
\]

Subject to workstation capacity constraints:
\[
\sum_{j=1}^{101} t_{1,j} x_j \leq 1,296
\]
\[
\sum_{j=1}^{101} t_{2,j} x_j \leq 1,238.4
\]
\[
\sum_{j=1}^{101} t_{3,j} x_j \leq 1,267.2
\]

Variable domains:
\[
x_j \in \mathbb{Z}_+, \quad \forall j = 1, \ldots, 101
\]

Where:
- \( t_{i,j} \) is the value in row \( i \), column \( j+1 \) of workstation_times.csv (with columns HiFi1_Minutes, ..., HiFi101_Minutes, in order).
- The mapping is: \( x_1 \) = HiFi1, \( x_2 \) = HiFi2, ..., \( x_{101} \) = HiFi101.

This is a complete numerical integer programming formulation, using all provided data, that minimizes total idle production time across the three workstations.