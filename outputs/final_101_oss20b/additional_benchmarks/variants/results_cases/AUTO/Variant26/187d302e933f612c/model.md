**Mathematical Model**

Let  

* \(W\) – set of workers (worker\_id)  
* \(P\) – set of projects (project\_id)  
* \(O \subseteq W\times P\) – set of *eligible offers*  
  (offers for which the worker is not on leave and whose skill level is
  at least the required skill of the project).

Parameters (all taken from the CSV files)

| Parameter | Definition | Source |
|-----------|------------|--------|
| `skill_rank[skill]` | Integer rank of a skill level: Junior = 1, Intermediate = 2, Senior = 3, Expert = 4 | Defined in the model |
| `worker_skill[w]` | Skill level of worker \(w\) | `file_0_view_0` (`skill`) |
| `on_leave[w]` | 1 if worker \(w\) is on leave, 0 otherwise | `file_0_view_0` (`on_leave`) |
| `required_skill[p]` | Required skill level for project \(p\) | `file_1_view_0` (`required_skill`) |
| `cost_cents[o]` | Cost in cents for offer \(o=(w,p)\) | `file_2_view_0` (`cost_cents`) |

**Decision Variables**

\[
x_{o}\in\{0,1\}\quad \forall o\in O
\]

\(x_{o}=1\) iff worker \(w\) is assigned to project \(p\).

**Objective**

\[
\min \sum_{o\in O} \text{cost_cents}[o]\;x_{o}
\]

**Constraints**

1. **Project coverage** – each project receives exactly one worker  

\[
\sum_{\substack{o=(w,p)\in O}} x_{o}=1 \quad \forall p\in P
\]

2. **Worker capacity** – each worker takes at most one project  

\[
\sum_{\substack{o=(w,p)\in O}} x_{o}\le 1 \quad \forall w\in W
\]

3. **Eligibility** – \(O\) contains only offers that satisfy  

   * \(on\_leave[w]=0\)  
   * \(\text{skill_rank}[worker\_skill[w]] \ge \text{skill_rank}[required\_skill[p]]\)

   (This is enforced by constructing \(O\) from the CSV data; no additional
   constraints are needed.)

---

**Data Mapping**

| Table ID | Columns | Meaning |
|----------|---------|---------|
| `file_0_view_0` | `worker_id`, `skill`, `on_leave` | Worker attributes |
| `file_1_view_0` | `project_id`, `required_skill` | Project requirements |
| `file_2_view_0` | `worker_id`, `project_id`, `cost_cents` | Offered assignment costs |

The set \(O\) is obtained by joining `file_2_view_0` with the worker and project tables and filtering on the conditions above. All indices and parameters are defined symbolically; numerical values are supplied by the CSV files at runtime.