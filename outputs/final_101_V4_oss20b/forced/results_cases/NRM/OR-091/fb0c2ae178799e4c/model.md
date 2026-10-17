**Mathematical Model**

Let  

- \(I\) – set of Operations Research courses (identified by `course_id`).  
- For each \(i\in I\):  
  - \(c_i\) – credits of course \(i\) (parameter).  
  - \(p_i\) – interest‑points of course \(i\) (parameter).  
- Decision variable  
  - \(x_i \in \{0,1\}\) – 1 if course \(i\) is selected, 0 otherwise.

\[
\begin{aligned}
\max\;& \sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} c_i\,x_i \;\le\; 20 \\[4pt]
& x_i \in \{0,1\}, \qquad \forall i\in I .
\end{aligned}
\]

---

**Data Mapping**

| Parameter / Variable | Source Table | Source Column |
|----------------------|--------------|---------------|
| \(I\) (index set)    | `file_0_view_0` | `course_id` |
| \(c_i\) (credits)    | `file_0_view_0` | `credits` |
| \(p_i\) (interest points) | `file_0_view_0` | `interest_points` |
| \(x_i\) (binary decision) | – | – |

*The table `file_0_view_0` is the filtered view of `courses_42.csv` containing only rows where `discipline = "Operations Research"`. No other data or numeric values are introduced beyond those mapped above.*