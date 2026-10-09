## Symbolic Mathematical Model

### Sets
- $W$: set of all workers, indexed by $w$ (corresponds to all columns except "Owner" in file_0_view_0)
- $H$: set of all homeowners, indexed by $h$ (corresponds to all rows, with "Owner" column giving the homeowner's worker ID)

### Parameters (from file_0_view_0)
- $d_{h,w}$: number of days worker $w$ worked on homeowner $h$'s home (from file_0_view_0, row $h$, column $w$)
- $w_0$: the first worker in $W$ (the first column after "Owner" in file_0_view_0, e.g., "Carpenter")
- $p_{w_0}$: fixed daily wage for $w_0$ ($60.00$ yuan)

### Decision Variables
- $p_w \geq 0$: daily wage for worker $w \in W$, with $p_{w_0} = 60.00$

### Model

**Balance constraints for each participant (homeowner/worker):**
For each $h \in H$ (with $h$'s own worker ID $w_h$ from "Owner" column):

$$
\sum_{\substack{w \in W \\ w \neq w_h}} d_{h,w} \cdot p_w = \sum_{\substack{h' \in H \\ h' \neq h}} d_{h',w_h} \cdot p_{w_h}
$$

- The left side is the total amount homeowner $h$ pays to all other workers for work done on their own home.
- The right side is the total income worker $w_h$ receives for working on other homes.

**Total work constraint for each worker:**
For each $w \in W$:
$$
\sum_{h \in H} d_{h,w} = 10
$$

**Wage normalization:**
$$
p_{w_0} = 60.00
$$

**Nonnegativity:**
$$
p_w \geq 0 \quad \forall w \in W
$$

### Data Mapping

- $W$: All columns except "Owner" in file_0_view_0
- $H$: All rows in file_0_view_0
- $d_{h,w}$: file_0_view_0, row $h$, column $w$
- $w_h$: file_0_view_0, row $h$, column "Owner"
- $w_0$: first column after "Owner" in file_0_view_0 (e.g., "Carpenter")
- $p_{w_0}$: fixed at $60.00$

### Summary

- For each participant, their total income from working on others' homes equals their total expenditure for work performed at their own home.
- Each worker's total work days (across all homes) is exactly 10.
- The daily wage of the first worker is fixed at 60.00 yuan.
- All other daily wages are determined to satisfy the above constraints.

This is a system of linear equations in the variables $\{p_w\}_{w \in W}$, with one variable fixed.