Let $x_k$ be the number of units of radio model HiFi-$k$ ($k=1,\ldots,101$) to produce per day. Let $t_{jk}$ be the processing time (in minutes) required for one unit of model $k$ at workstation $j$ ($j=1,2,3$), as given in the retrieved data. Each workstation $j$ has a total daily time of 1,440 minutes, but only a fraction is available for production due to maintenance: 90% for workstation 1, 86% for workstation 2, and 88% for workstation 3.

Define the effective daily capacity for each workstation:
- Workstation 1: $C_1 = 1,440 \times 0.90 = 1,296$ minutes
- Workstation 2: $C_2 = 1,440 \times 0.86 = 1,238.4$ minutes
- Workstation 3: $C_3 = 1,440 \times 0.88 = 1,267.2$ minutes

Let the idle time at workstation $j$ be $s_j = C_j - \sum_{k=1}^{101} t_{jk} x_k$.

The objective is to minimize the total idle time across all workstations:
\[
\min \sum_{j=1}^3 s_j = \sum_{j=1}^3 \left( C_j - \sum_{k=1}^{101} t_{jk} x_k \right)
\]
which is equivalent to:
\[
\max \sum_{j=1}^3 \sum_{k=1}^{101} t_{jk} x_k
\]
subject to the constraints below.

#### Mathematical Model

**Variables:**
- $x_k \in \mathbb{Z}_{\geq 0}$, for $k=1,\ldots,101$

**Parameters (from workstation_times.csv):**
- $t_{jk}$: Minutes required at workstation $j$ for one unit of HiFi-$k$ (see table below)
- $C_1 = 1,296$, $C_2 = 1,238.4$, $C_3 = 1,267.2$

**Objective:**
\[
\min \left[ (1,296 - \sum_{k=1}^{101} t_{1k} x_k) + (1,238.4 - \sum_{k=1}^{101} t_{2k} x_k) + (1,267.2 - \sum_{k=1}^{101} t_{3k} x_k) \right]
\]
or equivalently,
\[
\max \sum_{j=1}^3 \sum_{k=1}^{101} t_{jk} x_k
\]

**Subject to:**
\[
\sum_{k=1}^{101} t_{1k} x_k \leq 1,296
\]
\[
\sum_{k=1}^{101} t_{2k} x_k \leq 1,238.4
\]
\[
\sum_{k=1}^{101} t_{3k} x_k \leq 1,267.2
\]
\[
x_k \in \mathbb{Z}_{\geq 0} \quad \forall k=1,\ldots,101
\]

**Parameter Table (partial, see CSV for all 101 models):**

| Workstation | HiFi1_Minutes | HiFi2_Minutes | ... | HiFi101_Minutes | Maintenance_Percent |
|-------------|---------------|---------------|-----|-----------------|--------------------|
| 1           | 6             | 4             | ... | 9               | 10                 |
| 2           | 5             | 5             | ... | 3               | 14                 |
| 3           | 4             | 6             | ... | 6               | 12                 |

Use all $t_{jk}$ values as given in the retrieved data for $k=1,\ldots,101$ and $j=1,2,3$.

**Summary of Variables and Constraints:**
- Decision variables: $x_k$ (number of units of HiFi-$k$ to produce), integer, $\geq 0$
- For each workstation $j$, total processing time used $\leq$ effective capacity $C_j$
- Objective: minimize total idle time (equivalently, maximize total processing time used)

**All data and identifiers are used as retrieved and in original order.**