**Mathematical Model**

Let  

* \(W\) – set of workers (worker_id)  
* \(P\) – set of projects (project_id)

| Symbol | Meaning |
|--------|---------|
| \(x_{wp}\) | binary decision: 1 if worker \(w\) is assigned to project \(p\), 0 otherwise |
| \(c_{wp}\) | assignment cost in USD cents (defined only when a cost entry exists) |
| \(s_w\) | skill level of worker \(w\) (numeric) |
| \(r_p\) | required skill level of project \(p\) (numeric) |
| \(a_{wp}\) | 1 if assignment \(w\!\to\!p\) is allowed (cost exists **and** \(s_w\ge r_p\)), 0 otherwise |

Skill hierarchy (numeric encoding)  

| Skill | Code |
|-------|------|
| Junior | 1 |
| Intermediate | 2 |
| Senior | 3 |
| Expert | 4 |

---

### Objective
\[
\min \;\sum_{w\in W}\sum_{p\in P} c_{wp}\,x_{wp}
\]

### Constraints
1. **Project assignment** – each project gets exactly one worker  
   \[
   \sum_{w\in W} x_{wp}=1 \quad \forall p\in P
   \]

2. **Worker capacity** – each worker takes at most one project  
   \[
   \sum_{p\in P} x_{wp}\le 1 \quad \forall w\in W
   \]

3. **Feasibility** – disallow forbidden assignments  
   \[
   x_{wp}\le a_{wp}\quad \forall w\in W,\;p\in P
   \]

4. **Binary variables**  
   \[
   x_{wp}\in\{0,1\}\quad \forall w\in W,\;p\in P
   \]

---

**Data Mapping**

```json
{
  "workers": {
    "table_id": "file_0_view_0",
    "columns": ["worker_id", "skill", "on_leave"],
    "index": "worker_id",
    "parameters": {
      "s_w": {
        "source_column": "skill",
        "mapping": {
          "Junior": 1,
          "Intermediate": 2,
          "Senior": 3,
          "Expert": 4
        }
      }
    }
  },
  "projects": {
    "table_id": "file_1_view_0",
    "columns": ["project_id", "required_skill"],
    "index": "project_id",
    "parameters": {
      "r_p": {
        "source_column": "required_skill",
        "mapping": {
          "Junior": 1,
          "Intermediate": 2,
          "Senior": 3,
          "Expert": 4
        }
      }
    }
  },
  "costs": [
    {
      "table_id": "file_2_view_0",
      "columns": ["worker_id", "P00","P01","P02","P03","P04","P05","P06","P07","P08","P09"],
      "index": "worker_id",
      "cost_columns": ["P00","P01","P02","P03","P04","P05","P06","P07","P08","P09"]
    },
    {
      "table_id": "file_3_view_0",
      "columns": ["worker_id", "P00","P01","P02","P03","P04","P05","P06","P07","P08","P09"],
      "index": "worker_id",
      "cost_columns": ["P00","P01","P02","P03","P04","P05","P06","P07","P08","P09"]
    },
    {
      "table_id": "file_4_view_0",
      "columns": ["worker_id", "P00","P01","P02","P03","P04","P05","P06","P07","P08","P09"],
      "index": "worker_id",
      "cost_columns": ["P00","P01","P02","P03","P04","P05","P06","P07","P08","P09"]
    }
  ],
  "allowed": {
    "definition": "a_wp = 1 if c_wp exists AND s_w >= r_p, else 0"
  }
}
```

*All cost entries that are blank in the CSV files are treated as missing; such pairs receive `a_wp = 0` and the corresponding decision variable is forced to zero.*