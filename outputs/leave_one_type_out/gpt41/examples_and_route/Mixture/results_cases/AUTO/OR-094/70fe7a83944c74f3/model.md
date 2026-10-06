Let $x_k$ be the number of units of radio model $k$ (for $k = 1, \ldots, 101$) to produce per day. Let $t_{wk}$ be the processing time (in minutes) required for one unit of model $k$ at workstation $w$ (for $w = 1,2,3$). Let $C_w$ be the effective daily capacity (in minutes) of workstation $w$ after maintenance.

From the data:
- For each workstation, the total available time is 1,440 minutes.
- Maintenance percentages: workstation 1: 10%, workstation 2: 14%, workstation 3: 12%.
- Thus, effective capacities:
  - $C_1 = 1,440 \times (1 - 0.10) = 1,296$
  - $C_2 = 1,440 \times (1 - 0.14) = 1,238.4$
  - $C_3 = 1,440 \times (1 - 0.12) = 1,267.2$

Let $t_{wk}$ be the value in row $w$, column "HiFi$k$_Minutes" of workstation_times.csv.

Define the idle time at workstation $w$ as $I_w = C_w - \sum_{k=1}^{101} t_{wk} x_k$.

The objective is to minimize total idle time:
$$
\min \sum_{w=1}^3 I_w = \sum_{w=1}^3 \left( C_w - \sum_{k=1}^{101} t_{wk} x_k \right)
$$
which is equivalent to:
$$
\max \sum_{w=1}^3 \sum_{k=1}^{101} t_{wk} x_k
$$
subject to the constraints below.

#### Mathematical Model

**Decision variables:**
- $x_k \in \mathbb{Z}_{\geq 0}$, for $k = 1, \ldots, 101$

**Parameters:**
- $t_{wk}$: processing time (minutes) for model $k$ at workstation $w$ (from workstation_times.csv)
- $C_1 = 1,296$, $C_2 = 1,238.4$, $C_3 = 1,267.2$

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
x_k &\in \mathbb{Z}_{\geq 0} \quad \forall k = 1, \ldots, 101
\end{align*}
\]

**Where:**
- $t_{1k}$ = value in row Workstation=1, column HiFi$k$_Minutes
- $t_{2k}$ = value in row Workstation=2, column HiFi$k$_Minutes
- $t_{3k}$ = value in row Workstation=3, column HiFi$k$_Minutes

**All data from workstation_times.csv is used as follows:**

- For $k = 1, \ldots, 101$, the processing times $t_{1k}$, $t_{2k}$, $t_{3k}$ are taken from the corresponding columns (HiFi1_Minutes, ..., HiFi101_Minutes) in each row (Workstation 1, 2, 3).

**Summary:**

Minimize total idle time across all workstations by choosing integer production quantities $x_k$ for each model, subject to each workstation's effective daily capacity after maintenance, using the exact per-model, per-workstation processing times from the provided data.