**Mathematical Model**

Let  

- \(I\) be the set of Operations Research courses (identified by `course_id`).  
- For each \(i\in I\)  
  - \(c_i\) = number of credits of course \(i\).  
  - \(p_i\) = interest‑points score of course \(i\).  
- Decision variable  
  - \(x_i \in \{0,1\}\) – 1 if course \(i\) is selected, 0 otherwise.

\[
\begin{aligned}
\max \quad & \sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} c_i\,x_i \;\le\; 20 \\[4pt]
& x_i \in \{0,1\}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Mapping |
|--------|--------|---------|
| \(I\) | `file_0_view_0` | \(\{\,\text{course_id}\mid \text{discipline}=\text{“Operations Research”}\,\}\) |
| \(c_i\) | `file_0_view_0` | `credits` column for row \(i\) |
| \(p_i\) | `file_0_view_0` | `interest_points` column for row \(i\) |
| \(x_i\) | — | binary decision variable |

*Table identifier:* `file_0_view_0` (view of `courses_42.csv` filtered to Operations Research).  
*Columns used:* `course_id`, `credits`, `interest_points`.