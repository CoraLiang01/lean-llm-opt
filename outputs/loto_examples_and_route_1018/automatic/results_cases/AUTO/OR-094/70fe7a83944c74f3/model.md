Let $x_k$ be the number of units of radio model HiFi$k$ ($k=1,\ldots,101$) to produce per day. All $x_k$ are nonnegative integers.

Let $t_{jk}$ be the processing time (in minutes) required for one unit of model $k$ at workstation $j$ ($j=1,2,3$), as given in the table below.

Let $C_j$ be the total available time per day at workstation $j$ (1,440 minutes for all $j$).

Let $m_j$ be the maintenance percentage at workstation $j$ (10% for 1, 14% for 2, 12% for 3).

Let the effective capacity at workstation $j$ be $E_j = C_j \cdot (1 - m_j/100)$.

Define the idle time at workstation $j$ as $I_j = E_j - \sum_{k=1}^{101} t_{jk} x_k$.

The objective is to minimize total idle time across all workstations:
\[
\min \sum_{j=1}^3 I_j = \sum_{j=1}^3 \left[ E_j - \sum_{k=1}^{101} t_{jk} x_k \right]
\]
which is equivalent to:
\[
\max \sum_{j=1}^3 \sum_{k=1}^{101} t_{jk} x_k
\]
subject to the constraints below.

#### Data

- $C_1 = 1440$, $m_1 = 10$, $E_1 = 1296$
- $C_2 = 1440$, $m_2 = 14$, $E_2 = 1238.4$
- $C_3 = 1440$, $m_3 = 12$, $E_3 = 1267.2$

The processing times $t_{jk}$ are as follows (extracted from the CSV, shown for $k=1$ to $k=101$):

| Workstation | HiFi1 | HiFi2 | HiFi3 | HiFi4 | HiFi5 | ... | HiFi101 |
|-------------|-------|-------|-------|-------|-------|-----|---------|
| 1           | 6     | 4     | 6     | 7     | 6     | ... | 9       |
| 2           | 5     | 5     | 5     | 1     | 7     | ... | 3       |
| 3           | 4     | 6     | 5     | 2     | 6     | ... | 6       |

#### Mathematical Model

**Variables:**
\[
x_k \in \mathbb{Z}_{\geq 0} \quad \forall k=1,\ldots,101
\]

**Objective:**
\[
\min \left[ (1296 - \sum_{k=1}^{101} t_{1k} x_k) + (1238.4 - \sum_{k=1}^{101} t_{2k} x_k) + (1267.2 - \sum_{k=1}^{101} t_{3k} x_k) \right]
\]
or equivalently,
\[
\max \sum_{j=1}^3 \sum_{k=1}^{101} t_{jk} x_k
\]

**Constraints:**
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

**Where:**

- $t_{1k}$, $t_{2k}$, $t_{3k}$ are the processing times for model $k$ at workstations 1, 2, and 3, respectively, as given in the CSV:

(For clarity, here are the first few $t_{jk}$ values for each $k$:)

- $t_{1,1}=6$, $t_{1,2}=4$, $t_{1,3}=6$, ..., $t_{1,101}=9$
- $t_{2,1}=5$, $t_{2,2}=5$, $t_{2,3}=5$, ..., $t_{2,101}=3$
- $t_{3,1}=4$, $t_{3,2}=6$, $t_{3,3}=5$, ..., $t_{3,101}=6$

**Summary Table of Processing Times (first and last columns):**

| Model         | $t_{1k}$ | $t_{2k}$ | $t_{3k}$ |
|---------------|----------|----------|----------|
| HiFi1         | 6        | 5        | 4        |
| HiFi2         | 4        | 5        | 6        |
| ...           | ...      | ...      | ...      |
| HiFi101       | 9        | 3        | 6        |

**All $x_k$ are nonnegative integers.**