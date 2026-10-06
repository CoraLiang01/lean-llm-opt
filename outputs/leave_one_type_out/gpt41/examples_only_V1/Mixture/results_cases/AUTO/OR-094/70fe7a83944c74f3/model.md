Let $x_k$ be the number of units of radio model $k$ (for $k = 1, \ldots, 101$) to produce per day. Let $t_{ik}$ be the processing time (in minutes) required for one unit of model $k$ at workstation $i$ (for $i = 1,2,3$), as given in the table below. Each workstation $i$ has a total daily time of 1,440 minutes, but only a fraction is available for production due to maintenance: 90% for workstation 1, 86% for workstation 2, and 88% for workstation 3.

Define the effective daily capacities:
- Workstation 1: $C_1 = 1,440 \times 0.90 = 1,296$ minutes
- Workstation 2: $C_2 = 1,440 \times 0.86 = 1,238.4$ minutes
- Workstation 3: $C_3 = 1,440 \times 0.88 = 1,267.2$ minutes

Let the idle time at workstation $i$ be $s_i = C_i - \sum_{k=1}^{101} t_{ik} x_k$.

#### Decision Variables

- $x_k \in \mathbb{Z}_{\geq 0}$, for $k = 1, \ldots, 101$

#### Parameters

- $t_{ik}$: Processing time (in minutes) for one unit of model $k$ at workstation $i$, as given below.

#### Objective

Minimize total idle production time across all workstations:
$$
\min \sum_{i=1}^3 \left( C_i - \sum_{k=1}^{101} t_{ik} x_k \right)
$$

#### Constraints

For each workstation $i$:
$$
\sum_{k=1}^{101} t_{ik} x_k \leq C_i \qquad \forall i = 1,2,3
$$

And for all models:
$$
x_k \in \mathbb{Z}_{\geq 0} \qquad \forall k = 1, \ldots, 101
$$

#### Data

The processing times $t_{ik}$ are as follows (partial table shown for illustration; all 101 models and all 3 workstations are included in the data):

| Workstation | HiFi1_Minutes | HiFi2_Minutes | ... | HiFi101_Minutes | Maintenance_Percent |
|-------------|---------------|---------------|-----|-----------------|--------------------|
| 1           | 6             | 4             | ... | 9               | 10                 |
| 2           | 5             | 5             | ... | 3               | 14                 |
| 3           | 4             | 6             | ... | 6               | 12                 |

Where $t_{ik}$ is the value in row $i$, column $k+1$ (since the first column is Workstation).

#### Complete Model

Minimize:
$$
\left[1296 - \sum_{k=1}^{101} t_{1k} x_k\right] + \left[1238.4 - \sum_{k=1}^{101} t_{2k} x_k\right] + \left[1267.2 - \sum_{k=1}^{101} t_{3k} x_k\right]
$$

Subject to:
$$
\sum_{k=1}^{101} t_{1k} x_k \leq 1296 \\
\sum_{k=1}^{101} t_{2k} x_k \leq 1238.4 \\
\sum_{k=1}^{101} t_{3k} x_k \leq 1267.2 \\
x_k \in \mathbb{Z}_{\geq 0} \quad \forall k = 1, \ldots, 101
$$

Where $t_{ik}$ are the processing times from the retrieved data.