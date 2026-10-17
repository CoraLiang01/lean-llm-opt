#### Index Sets

- $T$: set of time periods (indexed by $t$), corresponding to all rows in 42.csv.
- $S$: set of possible shift start times (also indexed by $t$), where a shift can start at any time period.

#### Parameters

- $r_t$: required number of drivers and crew members in time period $t$ (from column "Number Required" in 42.csv, table_id: file_0_view_0).

#### Decision Variables

- $x_t$: number of drivers and crew members assigned to start work at the beginning of time period $t$ and work continuously for 4 hours. ($x_t \geq 0$, integer)

#### Objective

Minimize the total number of drivers and crew members assigned:
$$
\min \sum_{t \in S} x_t
$$

#### Constraints

For each time period $t \in T$:
$$
\sum_{s = t-3}^{t} x_{s \bmod |T|} \geq r_t \qquad \forall t \in T
$$

where $x_{s \bmod |T|}$ denotes that the index wraps around the 24-hour cycle (i.e., after period 24 comes period 1).

#### Variable Domains

$$
x_t \in \mathbb{Z}_{\geq 0} \qquad \forall t \in S
$$

---

#### Data Mapping

- Table: 42.csv (table_id: file_0_view_0)
    - Time period index $t$: row order in the file
    - Required number $r_t$: column "Number Required"
    - Time period label: column "Time"
    - Shift index: column "Shift" (used for mapping $t$ to time period)

All time periods and requirements are taken directly from the 24 rows of 42.csv, with no additional filtering.