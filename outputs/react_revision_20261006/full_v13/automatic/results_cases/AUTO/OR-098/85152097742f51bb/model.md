## Symbolic Mathematical Model

### Sets
- $W$: set of all workers, indexed by $w$ (columns of work_days.csv, excluding "Owner")
- $H$: set of all homeowners, indexed by $h$ (rows of work_days.csv, "Owner" column)

### Parameters (from Data Mapping)
- $d_{h,w}$: number of days worker $w$ worked on homeowner $h$'s home (from work_days.csv, table_id: file_0_view_0, columns: $w$, rows: $h$)
- $w_0$: the first worker in the file (e.g., "Carpenter")
- $p_{w_0}$: fixed daily wage for $w_0$ ($60.00$ yuan)
- $T$: total work days per worker ($10$)

### Decision Variables
- $p_w \geq 0$: daily wage of worker $w \in W$

### Objective
No explicit objective (feasibility system): determine $p_w$ for all $w \in W$ such that all constraints are satisfied.

### Constraints

1. **Fixed wage for first worker**
   $$
   p_{w_0} = 60.00
   $$

2. **Total work days per worker**
   $$
   \sum_{h \in H} d_{h,w} = 10 \qquad \forall w \in W
   $$

3. **Mutual payment balance for each participant**
   For each participant $i \in W$ (who is both a worker and a homeowner, i.e., $i$ appears in both $W$ and $H$):
   $$
   \sum_{\substack{h \in H \\ h \neq i}} d_{h,i} \, p_i = \sum_{w \in W} d_{i,w} \, p_w - d_{i,i} \, p_i
   \qquad \forall i \in W
   $$
   or equivalently,
   $$
   \sum_{h \in H} d_{h,i} \, p_i = \sum_{w \in W} d_{i,w} \, p_w \qquad \forall i \in W
   $$
   (since $d_{i,i} \, p_i$ appears on both sides).

   This enforces:  
   $[$Total income of $i$ from working on others' homes$]$ $=$ $[$Total payment by $i$ to all workers for their work on $i$'s home$]$

4. **Non-negativity**
   $$
   p_w \geq 0 \qquad \forall w \in W
   $$

### Data Mapping

- $d_{h,w}$: work_days.csv, table_id: file_0_view_0, columns: all except "Owner", rows: all
- $w_0$: first column after "Owner" in work_days.csv (e.g., "Carpenter")
- $p_{w_0}$: fixed at $60.00$
- $T$: $10$ (from problem statement)
- $W$: all columns except "Owner" in work_days.csv
- $H$: all values in "Owner" column in work_days.csv

### Summary

Find $p_w$ for all $w \in W$ such that:
- $p_{w_0} = 60.00$
- Each worker's total days worked is $10$
- For each participant $i$, $\sum_{h} d_{h,i} p_i = \sum_{w} d_{i,w} p_w$
- $p_w \geq 0$ for all $w$

All parameters and sets are mapped directly to the current work_days.csv (table_id: file_0_view_0).