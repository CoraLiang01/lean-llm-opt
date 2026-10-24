**Mathematical Model**

\[
\begin{aligned}
\text{minimize}\quad & \sum_{i\in I} c_i\,x_i \\[4pt]
\text{subject to}\quad & \sum_{i\in I} g_i\,x_i \;\ge\; 200 \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\end{aligned}
\]

**Index set**

- \(I\): set of all generation options (identified by the column **option**).

**Parameters**

- \(g_i\): generation per lot for option \(i\) (column **gen_per_lot**).
- \(c_i\): cost per lot for option \(i\) (column **cost_per_lot**).

**Decision variables**

- \(x_i\): number of lots of option \(i\) to purchase (integer, non‑negative).

---

### Data Mapping (CSVQA_DATA)

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | `file_0_view_0` | `option` |
| \(g_i\) | `file_0_view_0` | `gen_per_lot` |
| \(c_i\) | `file_0_view_0` | `cost_per_lot` |

All parameters are taken directly from the rows of `energy.csv` that satisfy the filter on `tech` (`coal`, `gas`, `renewables`). No additional data or constraints are introduced beyond those specified.