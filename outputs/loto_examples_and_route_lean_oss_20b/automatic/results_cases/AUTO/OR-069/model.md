**Mathematical Model**

```math
\begin{aligned}
\text{Sets}
&\ I = \{\,\text{Manager 1},\dots,\text{Manager 11}\,\},\\
&\ J = \{\,\text{Project 1},\dots,\text{Project 11}\,\}.\\[4pt]

\text{Parameters}
&\ c_{ij}\quad\text{(assignment cost of Manager }i\text{ to Project }j).\\[4pt]

\text{Decision Variables}
&\ x_{ij}\in\{0,1\}\quad\text{(1 if Manager }i\text{ is assigned to Project }j,\;0\text{ otherwise).}\\[4pt]

\text{Objective}
&\ \min \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}.\\[4pt]

\text{Constraints}
&\ \sum_{j\in J} x_{ij}=1 \qquad \forall i\in I \quad\text{(each manager gets one project)}\\
&\ \sum_{i\in I} x_{ij}=1 \qquad \forall j\in J \quad\text{(each project gets one manager)}\\
&\ x_{ij}\in\{0,1\}\qquad \forall i\in I,\;j\in J.
\end{aligned}
```

---

**Data Mapping Rules**

| Source | Mapping |
|--------|---------|
| **Table ID**: `file_0_view_0` | Provides all data for the model. |
| **Columns** | `Manager` → index `i` (manager identifier).<br>`Project 1 Cost` → `c_{i,Project 1}`.<br>`Project 2 Cost` → `c_{i,Project 2}`.<br>`…`<br>`Project 11 Cost` → `c_{i,Project 11}`. |
| **Rows** | Each row corresponds to a distinct manager; the order of rows in the CSV is preserved as the order of indices in set `I`. |
| **Parameter** | `c_{ij}` is read from the cell in row *i* and column `Project k Cost`, where `k` is the project number. |

These mappings ensure that every `c_{ij}` used in the objective and constraints is taken directly from the CSV file `manager_project_costs.csv` without any aggregation or alteration.