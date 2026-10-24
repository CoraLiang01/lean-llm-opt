**Mathematical Model**

Let  

* \(W\) – set of workers with `on_leave = 0`  
* \(P\) – set of all projects  

Parameters  

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(c_{wp}\) | Cost (USD cents) for worker \(w\) to perform project \(p\) (defined only if a non‑blank entry exists in any of the three cost files) | `file_2_view_0`, `file_3_view_0`, `file_4_view_0` |
| \(\text{skill\_rank}_w\) | Numeric rank of worker \(w\)’s skill (Junior = 1, Intermediate = 2, Senior = 3, Expert = 4) | `file_0_view_0` |
| \(\text{req\_rank}_p\) | Numeric rank of project \(p\)’s required skill | `file_1_view_0` |
| \(\delta_{wp}\) | 1 if \(c_{wp}\) is defined, 0 otherwise | derived from the three cost files |

Decision variables  

\[
x_{wp}\in\{0,1\}\quad\forall w\in W,\;p\in P
\]

---

### Objective
\[
\min \;\sum_{w\in W}\sum_{p\in P} c_{wp}\,x_{wp}
\]

### Constraints

1. **Each project is assigned to exactly one worker**  
   \[
   \sum_{w\in W} x_{wp}=1 \quad\forall p\in P
   \]

2. **Each worker takes at most one project**  
   \[
   \sum_{p\in P} x_{wp}\le 1 \quad\forall w\in W
   \]

3. **Skill compatibility**  
   \[
   \text{skill\_rank}_w \;\ge\; \text{req\_rank}_p \;-\; M\,(1-x_{wp}) \quad\forall w\in W,\;p\in P
   \]
   with \(M=4\) (maximum possible rank difference).

4. **Forbidden assignments**  
   \[
   x_{wp}\le \delta_{wp}\quad\forall w\in W,\;p\in P
   \]

---

**Data Mapping Rules**

| Parameter | Table ID | Column(s) | Mapping Rule |
|-----------|----------|-----------|--------------|
| \(c_{wp}\) | `file_2_view_0`, `file_3_view_0`, `file_4_view_0` | `worker_id`, `P00`–`P09` | For each pair \((w,p)\), take the first non‑blank value encountered in the order: `file_2_view_0` → `file_3_view_0` → `file_4_view_0`. |
| \(\delta_{wp}\) | Same as above | – | \(\delta_{wp}=1\) if a non‑blank value exists for \((w,p)\); otherwise 0. |
| \(\text{skill\_rank}_w\) | `file_0_view_0` | `worker_id`, `skill` | Map `Junior`→1, `Intermediate`→2, `Senior`→3, `Expert`→4. |
| \(\text{req\_rank}_p\) | `file_1_view_0` | `project_id`, `required_skill` | Same rank mapping as above. |
| \(W\) | `file_0_view_0` | `worker_id`, `on_leave` | Include worker \(w\) iff `on_leave` = 0. |
| \(P\) | `file_1_view_0` | `project_id` | All project IDs listed. |

All numeric values in the cost files are interpreted as **USD cents**; the objective value is therefore the minimum total cost in cents.