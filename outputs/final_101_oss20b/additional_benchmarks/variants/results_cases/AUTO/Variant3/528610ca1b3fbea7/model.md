## Mathematical Model

**Sets**  
\(I=\{A,B,\dots ,L\}\) – all activities (from *project_activities.csv*).  
\(P(i)\subseteq I\) – set of immediate predecessors of activity \(i\) (parsed from the
semicolon‑separated *Predecessors* field).

**Parameters**  
\(\text{ND}_i\) – normal duration of activity \(i\) (days).  
\(\text{CD}_i\) – crash duration of activity \(i\) (days).  
\(\text{CC}_i\) – crash cost per day of activity \(i\) (\$).  
\(\text{D}\) – project deadline (days) (from *project_parameters.csv*).  

**Decision Variables**  
\(s_i\ge 0\) – start time of activity \(i\) (continuous).  
\(z_i\in\mathbb{Z}_{\ge0}\) – integer number of days activity \(i\) is crashed.  
\(T\ge 0\) – project completion time (continuous).

**Objective**  
\[
\min \;\sum_{i\in I}\text{CC}_i\,z_i
\]

**Constraints**

1. **Crash bounds**  
   \[
   0\le z_i \le \text{ND}_i-\text{CD}_i ,\qquad \forall i\in I
   \]

2. **Precedence**  
   \[
   s_i \;\ge\; s_p + \bigl(\text{ND}_p - z_p\bigr),\qquad
   \forall i\in I,\;\forall p\in P(i)
   \]

3. **Activity completion**  
   \[
   T \;\ge\; s_i + \bigl(\text{ND}_i - z_i\bigr),\qquad \forall i\in I
   \]

4. **Project deadline**  
   \[
   T \;\le\; \text{D}
   \]

5. **Variable domains**  
   \[
   s_i\ge 0,\; T\ge 0,\; z_i\in\mathbb{Z}_{\ge0}\quad \forall i\in I
   \]

---

## Data Mapping

| Source | Table ID | Column(s) | Mapping to Model |
|--------|----------|-----------|------------------|
| *project_activities.csv* | file_0_view_0 | `Activity` | Index set \(I\) |
| | | `NormalDuration` | \(\text{ND}_i\) |
| | | `CrashDuration` | \(\text{CD}_i\) |
| | | `CrashCostPerDay` | \(\text{CC}_i\) |
| | | `Predecessors` | Parse into set \(P(i)\) (split on `;`) |
| *project_parameters.csv* | file_1_view_0 | `Value` where `Parameter`=`ProjectDeadline` | \(\text{D}\) |

All constraints and the objective use only these mapped parameters and the
decision variables defined above. No additional data or assumptions are introduced.