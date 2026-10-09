#### Mathematical Model

Let $x_s$ denote the number of drivers and crew members assigned to start work at the beginning of time period (shift) $s$, for each $s \in S$, where $S$ is the set of all Shift values in file_0_view_0.

Objective:
\[
\min \sum_{s \in S} x_s
\]

Subject to, for each $t \in S$:
\[
\sum_{k=0}^{3} x_{(t-k-1 \bmod |S|) + 1} \geq r_t
\]
where $r_t$ is the "Number Required" for shift $t$ (from file_0_view_0), and the indices are taken modulo $|S|$ (24), so that the 4-hour coverage wraps around midnight.

Variable domains:
\[
x_s \geq 0,\quad x_s \in \mathbb{Z},\quad \forall s \in S
\]

#### Data Mapping

- $S$: All Shift values from file_0_view_0, column "Shift"
- $r_t$: file_0_view_0, column "Number Required", indexed by Shift $t$
- $x_s$: Number of drivers and crew members assigned to start at Shift $s$ (decision variable)

- file_0_view_0: columns "Shift", "Time", "Number Required" (used for $S$ and $r_t$)

#### Notes

- Each $x_s$ represents the number of staff starting at the beginning of shift $s$.
- For each time period $t$, the sum covers $x_t$, $x_{t-1}$, $x_{t-2}$, $x_{t-3}$ (with wrap-around), ensuring every period is covered by staff who started in the last 4 periods.
- All variables are nonnegative integers.