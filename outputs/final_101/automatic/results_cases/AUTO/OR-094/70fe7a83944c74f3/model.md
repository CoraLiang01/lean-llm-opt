Let $x_k$ be the number of units of radio model HiFi-$k$ ($k=1,\ldots,101$) to produce per day. Let $t_{ik}$ be the processing time (in minutes) required for one unit of model $k$ at workstation $i$ ($i=1,2,3$), as given in the table below.

Define the effective daily capacity for each workstation:
- Workstation 1: $C_1 = 1440 \times (1 - 0.10) = 1296$ minutes
- Workstation 2: $C_2 = 1440 \times (1 - 0.14) = 1238.4$ minutes
- Workstation 3: $C_3 = 1440 \times (1 - 0.12) = 1267.2$ minutes

The idle time at workstation $i$ is $C_i - \sum_{k=1}^{101} t_{ik} x_k$.

Objective:
\[
\min \sum_{i=1}^3 \left[ C_i - \sum_{k=1}^{101} t_{ik} x_k \right]
\]
which is equivalent to
\[
\max \sum_{i=1}^3 \sum_{k=1}^{101} t_{ik} x_k
\]
subject to the constraints below.

Subject to:
\[
\sum_{k=1}^{101} t_{ik} x_k \leq C_i \qquad \forall i=1,2,3
\]
\[
x_k \in \mathbb{Z}_{\geq 0} \qquad \forall k=1,\ldots,101
\]

Where the $t_{ik}$ coefficients are as follows (from workstation_times.csv):

| Workstation | HiFi1_Minutes | HiFi2_Minutes | ... | HiFi101_Minutes | Maintenance_Percent |
|-------------|---------------|---------------|-----|-----------------|--------------------|
| 1           | 6             | 4             | ... | 9               | 10                 |
| 2           | 5             | 5             | ... | 3               | 14                 |
| 3           | 4             | 6             | ... | 6               | 12                 |

(Use all 101 HiFi columns as indexed by $k=1,\ldots,101$; the full table is as provided in the data.)

Explicitly:
- For workstation 1: $\sum_{k=1}^{101} t_{1k} x_k \leq 1296$
- For workstation 2: $\sum_{k=1}^{101} t_{2k} x_k \leq 1238.4$
- For workstation 3: $\sum_{k=1}^{101} t_{3k} x_k \leq 1267.2$
- $x_k \in \mathbb{Z}_{\geq 0}$ for all $k=1,\ldots,101$

All $t_{ik}$ coefficients are taken directly from the corresponding row and column in workstation_times.csv, preserving the original order and identifiers.

Minimize total idle time across all workstations by choosing the integer production quantities $x_k$ for each radio model.