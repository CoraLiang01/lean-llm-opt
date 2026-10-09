#### Abstract Mathematical Model for Mutual Payment of Wages

**Index Sets:**
- $W$: Set of all workers (columns in `work_days.csv` except "Owner")
- $H$: Set of all homeowners (rows in `work_days.csv`, each with a unique "Owner" value; $H = W$ by construction)

**Parameters:**
- $d_{h,w}$: Number of days worker $w \in W$ spent renovating homeowner $h \in H$'s home  
  (from `work_days.csv`, cell at row $h$, column $w$)
- $w_1$: The first worker in $W$ (i.e., the first column after "Owner" in the file)
- $r$: Fixed daily wage for $w_1$ (given as $60.00$ yuan)
- $T$: Total work days contributed by each worker (given as $10$)

**Variables:**
- $p_w$: Daily wage for worker $w \in W$, $p_{w_1}$ fixed at $r$, $p_w \in \mathbb{R}$ for $w \neq w_1$

**Objective:**
- No explicit objective: the model is a system of equations to determine a fair wage vector.

**Constraints:**

1. **Fairness (Balance) for Each Participant:**
   $$
   \sum_{\substack{h \in H \\ h \neq w}} d_{h,w} \cdot p_w = \sum_{w' \in W} d_{w,w'} \cdot p_{w'}, \quad \forall w \in W
   $$
   - Left: total income worker $w$ earns from working on others' homes.
   - Right: total payment worker $w$ owes for work done at their own home.

2. **Fixed Wage for First Worker:**
   $$
   p_{w_1} = r
   $$

3. **Total Work Days for Each Worker:**
   $$
   \sum_{h \in H} d_{h,w} = T, \quad \forall w \in W
   $$
   (This is a data property, not a constraint on $p_w$, but included for completeness.)

**Variable Domains:**
- $p_w \in \mathbb{R}$ for all $w \in W$; $p_{w_1}$ fixed at $r$.

---

**Data Mapping:**

- Table: `file_0_view_0` (from `work_days.csv`)
  - Index set $W$: All columns except "Owner"
  - Index set $H$: All rows, with "Owner" as the homeowner identifier
  - Parameter $d_{h,w}$: Value at row with "Owner" = $h$, column $w$
  - Fixed wage: $p_{w_1}$, where $w_1$ is the first worker column in the file (e.g., "Carpenter")
  - Total work days: $T$ (from user query, not a file column)

---

**Summary:**  
Solve for $p_w$ for all $w \in W$ (with $p_{w_1}$ fixed), such that for every participant, their total income from working on others' homes equals their total expenditure for work performed at their own home, using the work days matrix from `work_days.csv`.