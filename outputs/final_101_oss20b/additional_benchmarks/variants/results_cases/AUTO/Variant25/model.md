**Mathematical Model**

Let  

* \(W\) – set of workers with `on_leave = 0` (from *file_0_view_0*).  
* \(P\) – set of projects (from *file_1_view_0*).  

Define the skill level mapping  

\[
\text{lvl}(\text{Junior})=1,\;
\text{lvl}(\text{Intermediate})=2,\;
\text{lvl}(\text{Senior})=3,\;
\text{lvl}(\text{Expert})=4 .
\]

For each worker \(w\in W\) let  

\[
\text{skill}_w = \text{lvl}(\text{skill column of }w).
\]

For each project \(p\in P\) let  

\[
\text{req}_p = \text{lvl}(\text{required_skill column of }p).
\]

Let  

\[
c_{wp} = \text{cost entry in file_2_view_0 for worker }w\text{ and project }p
\]
(defined only when the cell is non‑blank; otherwise \(c_{wp}\) is undefined).

Define the binary decision variable  

\[
x_{wp}\in\{0,1\}\quad\forall w\in W,\;p\in P
\]

where \(x_{wp}=1\) iff worker \(w\) is assigned to project \(p\).

---

### Objective

\[
\min \;\sum_{w\in W}\sum_{p\in P} c_{wp}\,x_{wp}
\]

(The objective is in USD cents.)

---

### Constraints

1. **Project assignment** – each project gets exactly one worker  

   \[
   \sum_{w\in W} x_{wp}=1 \quad\forall p\in P
   \]

2. **Worker capacity** – each worker takes at most one project  

   \[
   \sum_{p\in P} x_{wp}\le 1 \quad\forall w\in W
   \]

3. **Feasibility of assignment** – assignment allowed only if  

   * worker is not on leave (already enforced by \(W\)),  
   * worker’s skill level is at least the project’s required level, and  
   * the cost matrix entry is present (non‑blank).  

   Introduce an auxiliary binary parameter  

   \[
   a_{wp}= 
   \begin{cases}
   1 & \text{if } \text{skill}_w \ge \text{req}_p \text{ and } c_{wp}\text{ exists}\\
   0 & \text{otherwise}
   \end{cases}
   \]

   Then enforce  

   \[
   x_{wp}\le a_{wp}\quad\forall w\in W,\;p\in P
   \]

---

**Data Mapping**

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(W\) | *file_0_view_0* | all `worker_id` where `on_leave = 0` |
| \(P\) | *file_1_view_0* | all `project_id` |
| \(\text{skill}_w\) | *file_0_view_0* | `lvl(skill)` |
| \(\text{req}_p\) | *file_1_view_0* | `lvl(required_skill)` |
| \(c_{wp}\) | *file_2_view_0* | value in column `project_id` for row `worker_id`; undefined if cell is blank |
| \(a_{wp}\) | Derived | 1 if `skill_w ≥ req_p` **and** `c_{wp}` defined; 0 otherwise |

All indices, parameters, and variables are defined symbolically; no numerical values are enumerated in the model.