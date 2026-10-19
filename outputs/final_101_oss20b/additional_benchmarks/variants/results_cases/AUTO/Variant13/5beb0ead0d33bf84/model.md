## Mathematical Model

**Indices**  
\(I\) – set of all activities (values of the *Activity* column).  
For each \(i\in I\),  
\(Pred(i)\subseteq I\) – set of predecessors of \(i\) (obtained from the *Predecessors* column, split by “;”; empty for activities with no predecessors).

**Parameters**  
\(\text{NormalDur}_i\) – normal duration of activity \(i\) (from *NormalDuration*).  
\(\text{CrashDur}_i\) – minimum (crashed) duration of activity \(i\) (from *CrashDuration*).  
\(\text{CrashCost}_i\) – cost per day of crashing activity \(i\) (from *CrashCostPerDay*).  
\(\text{MaxCrash}_i = \text{NormalDur}_i-\text{CrashDur}_i\).  
\(\text{Deadline}\) – project deadline (from *Value* where *Parameter* = “ProjectDeadline”).

**Decision Variables**  
\(s_i\ge 0\) – start time of activity \(i\) (continuous).  
\(z_i\in\mathbb{Z}_{\ge0}\) – number of crash days used for activity \(i\).  
\(T\ge 0\) – project completion time (continuous).

**Objective**  
\[
\min \;\sum_{i\in I}\text{CrashCost}_i\,z_i
\]

**Constraints**

1. **Crash‑day bounds**  
   \[
   0 \le z_i \le \text{MaxCrash}_i ,\qquad \forall i\in I
   \]

2. **Precedence (using crashed durations)**  
   \[
   s_i \;\ge\; s_p + \text{NormalDur}_p - z_p ,\qquad
   \forall i\in I,\;\forall p\in Pred(i)
   \]

3. **Project completion for each activity**  
   \[
   T \;\ge\; s_i + \text{NormalDur}_i - z_i ,\qquad \forall i\in I
   \]

4. **Project deadline**  
   \[
   T \;\le\; \text{Deadline}
   \]

5. **Non‑negativity**  
   \[
   s_i \ge 0,\; T \ge 0 ,\qquad \forall i\in I
   \]

6. **Integer requirement**  
   \[
   z_i \in \mathbb{Z}_{\ge0},\qquad \forall i\in I
   \]

---

## Data Mapping Rules

| Index / Parameter | Source | Column | Notes |
|-------------------|--------|--------|-------|
| \(I\) | `file_0_view_0` | `Activity` | All distinct activity identifiers |
| \(Pred(i)\) | `file_0_view_0` | `Predecessors` | Split by “;”; empty string → ∅ |
| \(\text{NormalDur}_i\) | `file_0_view_0` | `NormalDuration` | Numeric value |
| \(\text{CrashDur}_i\) | `file_0_view_0` | `CrashDuration` | Numeric value |
| \(\text{CrashCost}_i\) | `file_0_view_0` | `CrashCostPerDay` | Numeric value |
| \(\text{MaxCrash}_i\) | Derived | – | \(\text{NormalDur}_i-\text{CrashDur}_i\) |
| \(\text{Deadline}\) | `file_1_view_0` | `Value` | Row where `Parameter` = “ProjectDeadline” |

All variables and constraints are defined solely in terms of these mapped data; no additional numeric values are introduced.