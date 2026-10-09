#### Abstract Linear Programming Model

**Index Sets:**
- $T$: Set of all time intervals in the day (from the "Time" column in 44.csv), indexed by $t$.
- $S$: Set of all possible shift start times (also $T$), indexed by $s$.

**Parameters:**
- $r_t$: Minimum number of waitstaff required during time interval $t \in T$ (from "Requirement" column in 44.csv).
- $n$: Number of consecutive intervals in an 8-hour shift. (If each interval is 30 minutes, $n = 16$.)

**Decision Variables:**
- $x_s \in \mathbb{Z}_+, \quad \forall s \in S$: Number of waitstaff whose shift starts at time interval $s$.

**Objective:**
\[
\min \sum_{s \in S} x_s
\]

**Constraints:**

For every time interval $t \in T$:
\[
\sum_{s \in S: t \in \text{Shift}(s)} x_s \geq r_t
\]
where $\text{Shift}(s)$ is the set of $n$ consecutive intervals starting at $s$ (wrapping around midnight if necessary), i.e., all intervals covered by a shift starting at $s$.

**Variable Domains:**
\[
x_s \geq 0 \text{ and integer}, \quad \forall s \in S
\]

---

#### Data Mapping

- Table: 44.csv
    - Table ID: file_0_view_0
    - Index set $T$, $S$: "Time"
    - Parameter $r_t$: "Requirement"
    - All 48 records used, in source order.

- The mapping from shift start $s$ to covered intervals $\text{Shift}(s)$ is determined by the order of rows in 44.csv, with each shift covering $n$ consecutive intervals starting at $s$ (wrapping around as needed).