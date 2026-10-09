## Mathematical Model

**Sets**
- $I$: set of activities (from project_activities.csv)
- $P_i$: set of immediate predecessors of activity $i \in I$

**Parameters** (from Data Mapping below)
- $d_i^{\text{norm}}$: normal duration of activity $i$
- $d_i^{\text{crash}}$: minimum (crash) duration of activity $i$
- $c_i$: crash cost per day for activity $i$
- $D$: project deadline (from project_parameters.csv)

**Decision Variables**
- $s_i \geq 0$: start time of activity $i$ (continuous)
- $z_i \in \mathbb{Z}_{\geq 0}$: number of crash days used for activity $i$ (integer)
- $T \geq 0$: project completion time (continuous)

**Objective**
\[
\min \sum_{i \in I} c_i z_i
\]

**Constraints**

1. **Crash-day bounds** (integer crash days within limits)
   \[
   0 \leq z_i \leq d_i^{\text{norm}} - d_i^{\text{crash}} \qquad \forall i \in I
   \]

2. **Precedence constraints** (using crashed durations)
   \[
   s_i \geq s_j + d_j^{\text{norm}} - z_j \qquad \forall i \in I,\ \forall j \in P_i
   \]

3. **Project completion constraints**
   \[
   T \geq s_i + d_i^{\text{norm}} - z_i \qquad \forall i \in I
   \]

4. **Project deadline constraint**
   \[
   T \leq D
   \]

5. **Nonnegativity and integrality**
   \[
   s_i \geq 0 \qquad \forall i \in I
   \]
   \[
   z_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
   \]
   \[
   T \geq 0
   \]

---

### Data Mapping

- **Set $I$ (activities):** All "Activity" entries in project_activities.csv (table_id: file_0_view_0)
- **Predecessors $P_i$:** For each activity $i$, parse "Predecessors" column (split by ";" if multiple) in project_activities.csv (table_id: file_0_view_0)
- **$d_i^{\text{norm}}$:** "NormalDuration" column in project_activities.csv (table_id: file_0_view_0)
- **$d_i^{\text{crash}}$:** "CrashDuration" column in project_activities.csv (table_id: file_0_view_0)
- **$c_i$:** "CrashCostPerDay" column in project_activities.csv (table_id: file_0_view_0)
- **$D$:** "Value" where "Parameter" = "ProjectDeadline" in project_parameters.csv (table_id: file_1_view_0)

---

**All variables, sets, and parameters are defined directly from the provided CSV files as described above.**