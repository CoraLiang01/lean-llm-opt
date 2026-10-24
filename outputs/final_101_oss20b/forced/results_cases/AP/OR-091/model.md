**Mathematical Model**

Let  

- \(I\) be the set of Operations Research courses (indexed by their `course_id`).  
- For each \(i\in I\):  
  - \(c_i\) = credits of course \(i\)  
  - \(p_i\) = interest points of course \(i\)  
- Decision variable \(x_i\in\{0,1\}\) indicates whether course \(i\) is selected.

\[
\begin{aligned}
\max\;& \sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} c_i\,x_i \;\le\; 20 \\[4pt]
& x_i \in \{0,1\}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(c_i\)   | credits | `file_0_view_0` | `credits` |
| \(p_i\)   | interest_points | `file_0_view_0` | `interest_points` |
| Index \(i\) | course_id | `file_0_view_0` | `course_id` |

The index set \(I\) consists of all rows returned by the filter on `discipline = "Operations Research"` in `courses_42.csv`.