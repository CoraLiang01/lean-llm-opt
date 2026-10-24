**Mathematical Model**

Let  

- \(I\) – set of Operations Research courses (identified by `course_id`).  
- \(c_i\) – credits of course \(i\).  
- \(p_i\) – interest‑points of course \(i\).  
- \(y_i \in \{0,1\}\) – 1 if course \(i\) is selected, 0 otherwise.

\[
\begin{aligned}
\max \quad & \sum_{i\in I} p_i\,y_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} c_i\,y_i \;\le\; 20 \\[4pt]
& y_i \in \{0,1\}, \qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(c_i\) | Credits | `file_0_view_0` | `credits` |
| \(p_i\) | Interest points | `file_0_view_0` | `interest_points` |
| \(I\) | Course IDs | `file_0_view_0` | `course_id` |

The view `file_0_view_0` contains only rows where `discipline = "Operations Research"`. No additional constraints or capacities are required.