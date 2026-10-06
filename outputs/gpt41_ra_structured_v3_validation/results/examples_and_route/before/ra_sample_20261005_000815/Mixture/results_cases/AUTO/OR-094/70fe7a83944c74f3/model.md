Let $x_k$ be the number of units of radio model HiFi-$k$ ($k=1,\ldots,101$) to produce per day. Let $t_{wk}$ be the processing time (in minutes) required for one unit of model $k$ at workstation $w$ ($w=1,2,3$), as given in the table below. Each workstation $w$ has a total daily time of 1,440 minutes, but only a fraction is available for production after maintenance: 90% for workstation 1, 86% for workstation 2, and 88% for workstation 3.

Define the idle time at workstation $w$ as:
$$
\text{Idle}_w = \text{EffectiveCapacity}_w - \sum_{k=1}^{101} t_{wk} x_k
$$
where
\[
\text{EffectiveCapacity}_1 = 0.90 \times 1440 = 1296 \\
\text{EffectiveCapacity}_2 = 0.86 \times 1440 = 1238.4 \\
\text{EffectiveCapacity}_3 = 0.88 \times 1440 = 1267.2
\]

The objective is to minimize the total idle time across all workstations:
\[
\min \sum_{w=1}^3 \text{Idle}_w = \sum_{w=1}^3 \left( \text{EffectiveCapacity}_w - \sum_{k=1}^{101} t_{wk} x_k \right)
\]
which is equivalent to maximizing total processing time used:
\[
\max \sum_{w=1}^3 \sum_{k=1}^{101} t_{wk} x_k
\]
subject to the constraints below.

#### Mathematical Model

**Decision variables:**
\[
x_k \in \mathbb{Z}_{\geq 0} \quad \forall k=1,\ldots,101
\]

**Parameters:**
- $t_{wk}$: Minutes required at workstation $w$ for one unit of model $k$ (from the table below)
- $\text{EffectiveCapacity}_w$: Effective daily capacity at workstation $w$ (as above)

**Objective:**
\[
\min \sum_{w=1}^3 \left( \text{EffectiveCapacity}_w - \sum_{k=1}^{101} t_{wk} x_k \right)
\]

**Constraints:**
\[
\sum_{k=1}^{101} t_{wk} x_k \leq \text{EffectiveCapacity}_w \qquad \forall w=1,2,3
\]
\[
x_k \in \mathbb{Z}_{\geq 0} \qquad \forall k=1,\ldots,101
\]

**Data Table (partial, see full CSV for all 101 models):**

| Workstation | HiFi1_Minutes | HiFi2_Minutes | ... | HiFi101_Minutes | Maintenance_Percent |
|-------------|---------------|--------------|-----|-----------------|--------------------|
| 1           | 6             | 4            | ... | 9               | 10                 |
| 2           | 5             | 5            | ... | 3               | 14                 |
| 3           | 4             | 6            | ... | 6               | 12                 |

Where $t_{1k}$ is the value in row 1, column HiFi$k$_Minutes, $t_{2k}$ is from row 2, and $t_{3k}$ from row 3.

**Explicitly:**

For $w=1$:
\[
6x_1 + 4x_2 + 6x_3 + \cdots + 9x_{101} \leq 1296
\]
For $w=2$:
\[
5x_1 + 5x_2 + 5x_3 + \cdots + 3x_{101} \leq 1238.4
\]
For $w=3$:
\[
4x_1 + 6x_2 + 5x_3 + \cdots + 6x_{101} \leq 1267.2
\]

And $x_k \in \mathbb{Z}_{\geq 0}$ for all $k=1,\ldots,101$.

**Minimize:**
\[
(1296 - \sum_{k=1}^{101} t_{1k} x_k) + (1238.4 - \sum_{k=1}^{101} t_{2k} x_k) + (1267.2 - \sum_{k=1}^{101} t_{3k} x_k)
\]

**Subject to:**
\[
\sum_{k=1}^{101} t_{1k} x_k \leq 1296
\]
\[
\sum_{k=1}^{101} t_{2k} x_k \leq 1238.4
\]
\[
\sum_{k=1}^{101} t_{3k} x_k \leq 1267.2
\]
\[
x_k \in \mathbb{Z}_{\geq 0} \quad \forall k=1,\ldots,101
\]

All coefficients $t_{wk}$ are as given in the retrieved CSV data, preserving the original order and identifiers.