**Mathematical Model**

Let  

- \(I\) be the set of Operations Research courses (identified by `course_id`).  
- For each \(i\in I\)  
  - \(c_i\) = credits of course \(i\)  
  - \(p_i\) = interest‑points of course \(i\)  
- Decision variable  
  - \(x_i \in \{0,1\}\)  (1 if course \(i\) is selected, 0 otherwise)

\[
\begin{aligned}
\max\;& \sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} c_i\,x_i \;\le\; 20 \\[4pt]
& x_i \in \{0,1\}, \qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(c_i\)   | `file_0_view_0` | `credits` |
| \(p_i\)   | `file_0_view_0` | `interest_points` |
| Index \(i\) | `file_0_view_0` | `course_id` |

All values for the parameters are taken directly from the rows returned in the observation (the 7 Operations Research courses). No additional data or constraints are introduced.