#### Abstract Model for Mutual Payment of Wages in Multi-Party Renovation

**Index Sets:**
- $W$: Set of workers (indexed by $j$), corresponding to all columns except "Owner" in `file_0_view_0`.
- $H$: Set of homeowners (indexed by $i$), corresponding to all rows in `file_0_view_0`. Each $i$ is associated with a unique $j$ such that homeowner $i$ is worker $j$.

**Parameters:**
- $a_{ij}$: Number of days worker $j$ spent renovating homeowner $i$'s home. ($a_{ij}$ is the value in row $i$, column $j$ of `file_0_view_0`.)
- $w_1$: Fixed daily wage for the first worker listed in the file (column order), set to $60.00$ yuan.

**Variables:**
- $w_j$: Daily wage for worker $j \in W$, $w_j \geq 0$ (real, free except $w_1$ fixed).

**Constraints:**

1. **Wage Scale Fixing:**
   $$
   w_1 = 60.00
   $$

2. **Fair Mutual Payment for Each Participant:**
   For each $i \in H$ (homeowner $i$ is worker $i$):
   $$
   \sum_{j \in W,\, j \neq i} a_{ij} w_j = \sum_{k \in W,\, k \neq i} a_{ki} w_i
   $$
   - Left: Total payment by homeowner $i$ to others for work on their home.
   - Right: Total income of worker $i$ from working on others' homes.

3. **Total Work Days for Each Worker:**
   For each $j \in W$:
   $$
   \sum_{i \in H} a_{ij} = 10
   $$
   (This is a data property, not a constraint on $w_j$, but included for completeness.)

**Objective:**
- None (system of equations; the solution is the set of $w_j$ satisfying the above constraints).

---

#### Data Mapping

- **Source Table:** `file_0_view_0` (from `work_days.csv`)
- **Worker Set $W$:** All columns except "Owner"
- **Homeowner Set $H$:** All rows, with "Owner" column as identifier
- **Parameter $a_{ij}$:** Value at row $i$, column $j$
- **Fixed Wage:** $w_1$ corresponds to the first worker column in file order (e.g., "Carpenter"), fixed at $60.00$ yuan

---

**Note:** All data and identifiers are used as returned by the query, with no further filtering or abbreviation. The model is a system of linear equations for the unknown daily wages, with one wage fixed for scale.