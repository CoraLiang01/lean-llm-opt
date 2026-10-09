##### Objective Function:

$\min \sum_{(w,p) \in \mathcal{A}} \text{cost\_cents}_{w,p} \cdot x_{w,p}$

##### Constraints:

1. **Project Coverage:**  
$\sum_{\substack{w: (w,p) \in \mathcal{A}}} x_{w,p} = 1 \quad \forall p \in \mathcal{P}$

2. **Worker Assignment Limit:**  
$\sum_{\substack{p: (w,p) \in \mathcal{A}}} x_{w,p} \leq 1 \quad \forall w \in \mathcal{W}$

3. **Skill Feasibility:**  
$x_{w,p} = 0$ if $\text{skill}_w < \text{required\_skill}_p$ (with Junior $<$ Intermediate $<$ Senior $<$ Expert)

4. **Leave Exclusion:**  
$x_{w,p} = 0$ if $\text{on\_leave}_w = 1$

5. **Offer List Restriction:**  
$x_{w,p}$ defined only for $(w,p) \in \mathcal{A}$, where $\mathcal{A}$ is the set of all $(w,p)$ pairs listed in the offer list.

6. **Variable Domain:**  
$x_{w,p} \in \{0,1\} \quad \forall (w,p) \in \mathcal{A}$

##### Data Mapping

- $\mathcal{W}$: Set of worker_id from `file_0_view_0` with on_leave = 0.
- $\mathcal{P}$: Set of project_id from `file_1_view_0`.
- $\mathcal{A}$: Set of $(worker_id, project_id)$ pairs from `file_2_view_0` where worker_id $\in \mathcal{W}$.
- $\text{cost\_cents}_{w,p}$: Value from `cost_cents` column in `file_2_view_0` for $(w,p) \in \mathcal{A}$.
- $\text{skill}_w$: Value from `skill` column in `file_0_view_0` for $w$.
- $\text{required\_skill}_p$: Value from `required_skill` column in `file_1_view_0` for $p$.
- $\text{on\_leave}_w$: Value from `on_leave` column in `file_0_view_0` for $w$.

##### Notes

- The minimum is reported in USD cents.
- Skill levels are ordered: Junior $<$ Intermediate $<$ Senior $<$ Expert.
- Only assignments listed in the offer list and meeting all constraints are permitted.