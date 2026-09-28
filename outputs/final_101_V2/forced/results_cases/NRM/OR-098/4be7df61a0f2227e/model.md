#### Abstract Mathematical Model for Mutual Payment of Wages

**Index Sets:**
- $W$: Set of all workers (indexed by $j$), corresponding to all columns except "Owner" in `file_0_view_0` (work_days.csv).
- $H$: Set of all homeowners (indexed by $i$), corresponding to all rows in `file_0_view_0` (work_days.csv). Each $i$ is associated with a unique $j$ such that worker $j$ is the owner of home $i$.

**Parameters:**
- $a_{ij}$: Number of days worker $j$ spent renovating homeowner $i$'s home. From column $j$ and row $i$ in `file_0_view_0`.
- $j_0$: The first worker listed in the file (e.g., "Carpenter").
- $r_{j_0}$: Fixed daily wage for worker $j_0$ (given as 60.00 yuan).
- $T$: Total work days contributed by each worker (given as 10).

**Variables:**
- $r_j$: Daily wage for worker $j$, $\forall j \in W$, $j \neq j_0$.

**Objective:**
- None (the model is a system of equations to determine fair wages).

**Constraints:**

1. **Fairness (Balance) Constraints:**  
   For every participant $k \in W$ (who is also a homeowner $i_k$), the total income from working on others' homes equals the total expenditure for work performed at their own home:
   $$
   \sum_{\substack{i \in H \\ i \neq i_k}} a_{i k} \cdot r_k = \sum_{j \in W} a_{i_k j} \cdot r_j, \quad \forall k \in W
   $$
   - $i_k$ is the row where $Owner = k$.

2. **Fixed Wage Constraint:**  
   $$
   r_{j_0} = 60.00
   $$

3. **Total Work Days Constraint (data property, not a model constraint):**  
   $$
   \sum_{i \in H} a_{i j} = T, \quad \forall j \in W
   $$
   (This is a property of the data, not a constraint to enforce.)

**Variable Domains:**
- $r_j \in \mathbb{R}$, $\forall j \in W$

---

**Data Mapping:**

- Table: `file_0_view_0` (from work_days.csv)
    - Index set $W$: All columns except "Owner"
    - Index set $H$: All rows
    - Parameter $a_{ij}$: Entry in row $i$, column $j$
    - Mapping between $i$ and $j$: For each row, the "Owner" column gives the worker $j$ who owns home $i$
    - Fixed wage: $r_{j_0}$, where $j_0$ is the first worker column in the file
    - Total work days: sum of each column $j$ over all rows $i$

---

**Summary:**  
This model determines the daily wage for each worker so that, for every participant, their total income from working on others' homes equals their total expenditure for work performed at their own home, with the daily wage of the first worker fixed at 60.00 yuan. All data is mapped directly from `file_0_view_0` (work_days.csv).