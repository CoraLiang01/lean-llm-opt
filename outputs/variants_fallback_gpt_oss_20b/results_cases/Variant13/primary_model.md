## Mathematical Model

**Sets**  
\(I=\{A,B,C,D,E,F,G,H,I,J\}\) – all activities.  
\(P(i)\subseteq I\) – set of predecessors of activity \(i\) (obtained from the *Predecessors* column).

**Parameters**  
\(\text{ND}_i\) – normal duration of activity \(i\).  
\(\text{CD}_i\) – crash‑duration (minimum possible duration) of activity \(i\).  
\(\text{CC}_i\) – crash‑cost per day for activity \(i\).  
\(\text{DL}\) – project deadline (value from *ProjectDeadline*).

**Decision Variables**  
\(s_i\ge 0\) – start time of activity \(i\) (continuous).  
\(z_i\in\mathbb{Z}_{\ge0}\) – number of crash days used for activity \(i\).  
\(T\ge 0\) – project completion time (continuous).

**Objective**  
\[
\min \;\sum_{i\in I}\text{CC}_i\,z_i
\]

**Constraints**

1. **Crash‑day bounds**  
   \[
   0\le z_i \le \text{ND}_i-\text{CD}_i ,\qquad \forall i\in I
   \]

2. **Precedence**  
   \[
   s_i \;\ge\; s_p + \text{ND}_p - z_p ,\qquad \forall i\in I,\;\forall p\in P(i)
   \]

3. **Activity completion**  
   \[
   T \;\ge\; s_i + \text{ND}_i - z_i ,\qquad \forall i\in I
   \]

4. **Project deadline**  
   \[
   T \;\le\; \text{DL}
   \]

5. **Variable domains**  
   \[
   s_i\ge 0,\; T\ge 0,\; z_i\in\mathbb{Z}_{\ge0}\quad \forall i\in I
   \]

---

## Data Mapping

| Parameter / Variable | Source Table | Column(s) | Notes |
|----------------------|--------------|-----------|-------|
| \(\text{ND}_i\) | `project_activities.csv` | `NormalDuration` | one value per activity |
| \(\text{CD}_i\) | `project_activities.csv` | `CrashDuration` | one value per activity |
| \(\text{CC}_i\) | `project_activities.csv` | `CrashCostPerDay` | one value per activity |
| \(P(i)\) | `project_activities.csv` | `Predecessors` | comma‑separated list; split into set |
| \(\text{DL}\) | `project_parameters.csv` | `Value` where `Parameter`=`ProjectDeadline` | single scalar |
| \(s_i\) | decision variable | – | continuous, \(\ge0\) |
| \(z_i\) | decision variable | – | integer, \(\ge0\) |
| \(T\) | decision variable | – | continuous, \(\ge0\) |

All constraints and the objective use only the parameters and variables listed above, with no additional data introduced. The model is a mixed‑integer linear program (MILP) that minimizes total crashing cost while respecting precedence, crash limits, and the project deadline.