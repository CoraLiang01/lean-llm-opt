## Mathematical Model

**Sets**
- $T$: set of time intervals, indexed by $t$ (from 1 to 48, each representing a 30-minute interval; see Data Mapping for exact labels)
- $S$: set of possible shift start times, indexed by $s$ (also 1 to 48, one for each interval)

**Parameters**
- $r_t$: minimum number of waitstaff required in interval $t$ (from column "Requirement", table_id: file_0_view_0)
- $L$: number of consecutive intervals in a shift ($L = 16$, since 8 hours = 16 half-hour intervals)

**Decision Variables**
- $x_s \in \mathbb{Z}_+, \forall s \in S$: number of waitstaff whose shift starts at interval $s$

**Objective**
$$
\min \sum_{s \in S} x_s
$$

**Constraints**
For each interval $t \in T$:
$$
\sum_{s \in S: \ t \in \{s, s+1, \ldots, s+L-1\} \pmod{48}} x_s \geq r_t
$$

where the sum is over all $s$ such that interval $t$ is covered by the 8-hour shift starting at $s$, with wrap-around at midnight (i.e., modulo 48).

**Variable Domains**
$$
x_s \geq 0, \quad x_s \in \mathbb{Z}, \quad \forall s \in S
$$

---

### Data Mapping

- $T$, $S$: All 48 intervals from column "Time" in table_id: file_0_view_0, with $t=1$ corresponding to "2:00am - 2:30am", $t=2$ to "2:30am - 3:00am", ..., $t=48$ to "1:30am - 2:00am".
- $r_t$: "Requirement" column in table_id: file_0_view_0, for each $t$.
- $L = 16$: Each shift covers 16 consecutive intervals (8 hours).
- $x_s$: Number of waitstaff starting at interval $s$.

All data is from table_id: file_0_view_0, columns "Time" and "Requirement".