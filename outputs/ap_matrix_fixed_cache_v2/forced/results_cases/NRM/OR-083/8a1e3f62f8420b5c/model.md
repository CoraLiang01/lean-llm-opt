#### Index Sets

- $W$: set of workers (from column "Task Time Required" in 15.csv, excluding header row; 12 workers)
- $T$: set of tasks (from columns "A" through "J" in 15.csv; 10 tasks)

#### Parameters

- $c_{w,t}$: time required for worker $w \in W$ to complete task $t \in T$ (from 15.csv, table_id: file_0_view_0, columns "A"–"J", rows indexed by "Task Time Required")

#### Decision Variables

- $x_{w,t} \in \{0,1\}$: 1 if worker $w$ is assigned to task $t$, 0 otherwise
- $y_w \in \{0,1\}$: 1 if worker $w$ is selected (assigned to any task), 0 otherwise

#### Objective

$$
\min \sum_{w \in W} \sum_{t \in T} c_{w,t} \cdot x_{w,t}
$$

#### Constraints

1. **Each task is assigned to exactly one worker:**
   $$
   \sum_{w \in W} x_{w,t} = 1 \quad \forall t \in T
   $$

2. **Each selected worker is assigned to exactly one task (and unselected workers are assigned to none):**
   $$
   \sum_{t \in T} x_{w,t} = y_w \quad \forall w \in W
   $$

3. **Exactly 10 workers are selected:**
   $$
   \sum_{w \in W} y_w = 10
   $$

4. **Variable domains:**
   $$
   x_{w,t} \in \{0,1\} \quad \forall w \in W,\, t \in T
   $$
   $$
   y_w \in \{0,1\} \quad \forall w \in W
   $$

---

#### Data Mapping

- **Source:** 15.csv, table_id: file_0_view_0
- **Worker index set $W$:** All rows with "Task Time Required" values 1–12 (excluding header)
- **Task index set $T$:** All columns "A" through "J"
- **Parameter $c_{w,t}$:** Value at row with "Task Time Required" = $w$, column $t$ ("A"–"J")
- **No additional filters applied; all data used as returned by CSVQA**