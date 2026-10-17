#### Abstract Linear Programming Model

**Index Sets:**
- $T$: Set of all time periods (half-hour intervals), as indexed in 44.csv.
- $S$: Set of all possible shift start times (one for each time period in $T$).

**Parameters:**
- $r_t$: Minimum number of waitstaff required in time period $t \in T$.  
  (From column "Requirement" in 44.csv)
- $C_{s,t}$: Coverage indicator, $C_{s,t} = 1$ if a shift starting at $s \in S$ covers time period $t \in T$, $0$ otherwise.  
  (A shift starting at $s$ covers the 16 consecutive half-hour periods starting at $s$, wrapping around midnight as needed.)

**Decision Variables:**
- $x_s \in \mathbb{Z}_+, \quad \forall s \in S$  
  Number of waitstaff starting a shift at time $s$.

**Objective:**
$$
\min \sum_{s \in S} x_s
$$

**Constraints:**
- For every time period $t \in T$:
$$
\sum_{s \in S} C_{s,t} \cdot x_s \geq r_t
$$

- $x_s \geq 0$ and integer, $\forall s \in S$

---

#### Data Mapping

- Table: 44.csv (table_id: file_0_view_0)
    - Index set $T$: All rows, column "Time"
    - Parameter $r_t$: Column "Requirement"
- Coverage matrix $C_{s,t}$: For each $s \in S$, $C_{s,t} = 1$ if $t$ is within the 8-hour (16 half-hour) window starting at $s$, $0$ otherwise. Both $S$ and $T$ are the set of all time periods in the file.

No additional data sources or filters were applied. All 48 time periods and requirements are used as returned.