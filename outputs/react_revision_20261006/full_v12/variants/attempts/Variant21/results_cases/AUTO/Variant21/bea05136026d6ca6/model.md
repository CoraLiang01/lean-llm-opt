## Mathematical Model

**Sets**
- $N$: set of activities (from project_activities.csv), indexed by $i$
- $P_i$: set of immediate predecessors of activity $i$ (from Predecessors column; $P_i = \emptyset$ if blank)

**Parameters** (from project_activities.csv and project_parameters.csv; see Data Mapping)
- $d_i^{\text{norm}}$: normal duration of activity $i$
- $d_i^{\text{crash}}$: minimum (crash) duration of activity $i$
- $c_i$: crash cost per day for activity $i$
- $D$: project deadline (from project_parameters.csv)

**Variables**
- $s_i \geq 0$: start time of activity $i$ (continuous)
- $z_i \in \mathbb{Z},\ 0 \leq z_i \leq d_i^{\text{norm}} - d_i^{\text{crash}}$: number of crash days used for activity $i$ (integer)
- $T \geq 0$: project completion time (continuous)

**Objective**
\[
\min \sum_{i \in N} c_i\, z_i
\]

**Constraints**

1. **Precedence constraints** (for all $i \in N$, for all $j \in P_i$):
   \[
   s_i \geq s_j + d_j^{\text{norm}} - z_j
   \]

2. **Crash-day bounds** (for all $i \in N$):
   \[
   0 \leq z_i \leq d_i^{\text{norm}} - d_i^{\text{crash}}
   \]
   \[
   z_i \in \mathbb{Z}
   \]

3. **Project completion constraints** (for all $i \in N$):
   \[
   T \geq s_i + d_i^{\text{norm}} - z_i
   \]

4. **Project deadline constraint**:
   \[
   T \leq D
   \]

5. **Nonnegativity** (for all $i \in N$):
   \[
   s_i \geq 0,\quad T \geq 0
   \]

---

### Data Mapping

- **Activities and Predecessors**: $N$ and $P_i$ from `project_activities.csv` columns "Activity" and "Predecessors" (split by ";" if multiple).
- **Durations and Costs**: $d_i^{\text{norm}}$ = "NormalDuration", $d_i^{\text{crash}}$ = "CrashDuration", $c_i$ = "CrashCostPerDay" from `project_activities.csv`.
- **Project Deadline**: $D$ = "Value" where "Parameter" = "ProjectDeadline" from `project_parameters.csv`.