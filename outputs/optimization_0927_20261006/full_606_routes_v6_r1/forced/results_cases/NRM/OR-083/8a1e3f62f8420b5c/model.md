#### Index Sets

- $W$: set of workers (from column "Task Time Required" in file_0_view_0 of 15.csv)
- $T$: set of tasks (from columns "A" through "J" in file_0_view_0 of 15.csv)

#### Parameters

- $c_{w,t}$: time required for worker $w \in W$ to complete task $t \in T$ (from file_0_view_0, 15.csv, columns "A"–"J", rows with worker indices)

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

- **file_0_view_0, 15.csv**
  - Worker set $W$: values in column "Task Time Required" (excluding header row)
  - Task set $T$: columns "A" through "J"
  - Parameter $c_{w,t}$: value at row with worker $w$ and column $t$
  - All rows and columns used; no filters applied

---

This model selects 10 out of 12 workers and assigns each to exactly one task, with each task assigned to exactly one worker, minimizing total working hours.