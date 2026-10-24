**Mathematical Model**

Let  

- \(I\) – set of Operations Research courses (identified by `course_id`).  
- For each \(i\in I\):  
  - \(c_i\) – credits of course \(i\) (parameter).  
  - \(p_i\) – interest‑points of course \(i\) (parameter).  
- Decision variable  
  \[
  x_i \in \{0,1\}\quad\forall i\in I
  \]
  (1 if course \(i\) is selected, 0 otherwise).

\[
\begin{aligned}
\max\;& \sum_{i\in I} p_i\,x_i \\
\text{s.t. } & \sum_{i\in I} c_i\,x_i \;\le\; 20 \\
& x_i \in \{0,1\}\quad\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Element | Source | Details |
|---------|--------|---------|
| **Table** | `file_0_view_0` | View of `courses_42.csv` filtered by `discipline = "Operations Research"` |
| **Index set \(I\)** | `course_id` column | Each distinct `course_id` in the filtered view |
| **Parameter \(c_i\)** | `credits` column | Integer credits for course \(i\) |
| **Parameter \(p_i\)** | `interest_points` column | Integer interest points for course \(i\) |
| **Variable \(x_i\)** | – | Binary decision variable for course \(i\) |
| **Constraint** | – | Total credits ≤ 20 |
| **Objective** | – | Maximize total interest points |

All data values are taken directly from the filtered rows of `file_0_view_0`; no additional aggregation or transformation is performed.