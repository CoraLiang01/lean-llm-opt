#### Abstract Linear Programming Model

**Index Sets:**
- $T$: set of time periods (from 42.csv, column "Shift" or "Time"), $T = \{1, 2, ..., n\}$, where $n$ is the number of time periods.

**Parameters:**
- $r_t$: number of drivers and crew members required in time period $t \in T$ (from 42.csv, column "Number Required").

**Decision Variables:**
- $x_t$: number of drivers and crew members assigned to start work at the beginning of time period $t$, $x_t \geq 0$, integer, $\forall t \in T$.

**Objective:**
\[
\min \sum_{t \in T} x_t
\]

**Constraints:**

For each time period $t \in T$:
\[
\sum_{k = \max\{1, t-3\}}^{t} x_k \geq r_t
\]
(That is, the sum of assignments starting in the current or previous three periods must cover the requirement in period $t$, since each assignment lasts 4 consecutive periods.)

**Variable Domains:**
\[
x_t \in \mathbb{Z}_+, \quad \forall t \in T
\]

---

#### Data Mapping

- Table: 42.csv (table_id: file_0_view_0)
    - Index set $T$: column "Shift" or "Time"
    - Parameter $r_t$: column "Number Required"

No literal values or record counts are included in the abstract model.