Let $x_k$ be the number of units of radio model HiFi-$k$ ($k=1,\ldots,101$) to produce per day. Let $t_{ik}$ be the processing time (in minutes) required for one unit of model $k$ at workstation $i$ ($i=1,2,3$), as given in the table below. Each workstation $i$ has a total daily time of 1,440 minutes, but only a fraction is available for production due to maintenance: 90% for workstation 1, 86% for workstation 2, and 88% for workstation 3.

Define the effective daily capacities:
- Workstation 1: $C_1 = 1,440 \times 0.90 = 1,296$ minutes
- Workstation 2: $C_2 = 1,440 \times 0.86 = 1,238.4$ minutes
- Workstation 3: $C_3 = 1,440 \times 0.88 = 1,267.2$ minutes

Let the idle time at workstation $i$ be $s_i = C_i - \sum_{k=1}^{101} t_{ik} x_k$.

The objective is to minimize total idle time across all workstations:
$$
\min \sum_{i=1}^3 s_i = \sum_{i=1}^3 \left(C_i - \sum_{k=1}^{101} t_{ik} x_k\right)
$$
which is equivalent to:
$$
\max \sum_{i=1}^3 \sum_{k=1}^{101} t_{ik} x_k
$$
subject to the constraints below.

#### Model

**Variables:**
- $x_k \in \mathbb{Z}_{\geq 0}$, for $k=1,\ldots,101$

**Parameters (from workstation_times.csv):**

| Workstation | HiFi1_Minutes | HiFi2_Minutes | ... | HiFi101_Minutes | Maintenance_Percent |
|-------------|---------------|---------------|-----|-----------------|--------------------|
| 1           | 6             | 4             | ... | 10              | 10                 |
| 2           | 5             | 5             | ... | 3               | 14                 |
| 3           | 4             | 6             | ... | 6               | 12                 |

**Objective:**
$$
\min \left[ (1,296 - \sum_{k=1}^{101} t_{1k} x_k) + (1,238.4 - \sum_{k=1}^{101} t_{2k} x_k) + (1,267.2 - \sum_{k=1}^{101} t_{3k} x_k) \right]
$$

**Constraints:**
\[
\begin{align*}
\sum_{k=1}^{101} t_{1k} x_k &\leq 1,296 \\
\sum_{k=1}^{101} t_{2k} x_k &\leq 1,238.4 \\
\sum_{k=1}^{101} t_{3k} x_k &\leq 1,267.2 \\
x_k &\in \mathbb{Z}_{\geq 0} \quad \forall k=1,\ldots,101
\end{align*}
\]

**Where:**
- $t_{ik}$ is the processing time (in minutes) for one unit of model $k$ at workstation $i$, as given in the table below (from workstation_times.csv, source order preserved):

| Workstation | HiFi1_Minutes | HiFi2_Minutes | ... | HiFi101_Minutes | Maintenance_Percent |
|-------------|---------------|---------------|-----|-----------------|--------------------|
| 1           | 6             | 4             | ... | 10              | 10                 |
| 2           | 5             | 5             | ... | 3               | 14                 |
| 3           | 4             | 6             | ... | 6               | 12                 |

**Decision variables:**
- $x_1$: units of HiFi-1 to produce per day
- $x_2$: units of HiFi-2 to produce per day
- $\vdots$
- $x_{101}$: units of HiFi-101 to produce per day

**Summary of retrieved data:**
- Workstation 1: $t_{1,1}=6$, $t_{1,2}=4$, ..., $t_{1,101}=10$, maintenance 10%
- Workstation 2: $t_{2,1}=5$, $t_{2,2}=5$, ..., $t_{2,101}=3$, maintenance 14%
- Workstation 3: $t_{3,1}=4$, $t_{3,2}=6$, ..., $t_{3,101}=6$, maintenance 12%

**Complete Model:**

Minimize
$$
(1,296 - \sum_{k=1}^{101} t_{1k} x_k) + (1,238.4 - \sum_{k=1}^{101} t_{2k} x_k) + (1,267.2 - \sum_{k=1}^{101} t_{3k} x_k)
$$

Subject to
\[
\begin{align*}
\sum_{k=1}^{101} t_{1k} x_k &\leq 1,296 \\
\sum_{k=1}^{101} t_{2k} x_k &\leq 1,238.4 \\
\sum_{k=1}^{101} t_{3k} x_k &\leq 1,267.2 \\
x_k &\in \mathbb{Z}_{\geq 0} \quad \forall k=1,\ldots,101
\end{align*}
\]

All coefficients $t_{ik}$ are as given in the retrieved workstation_times.csv, source order preserved.