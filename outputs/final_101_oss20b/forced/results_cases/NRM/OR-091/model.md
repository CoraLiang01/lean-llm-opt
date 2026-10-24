**Mathematical Model**

Let  

- \(I\) be the set of Operations Research courses (identified by `course_id`).  
- For each \(i\in I\)  
  - \(c_i\)  – credits of course \(i\) (from column **credits**).  
  - \(p_i\)  – interest‑points of course \(i\) (from column **interest_points**).  
- Decision variable  
  - \(x_i \in \{0,1\}\) – 1 if course \(i\) is selected, 0 otherwise.

\[
\begin{aligned}
\max\;& \sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} c_i\,x_i \;\le\; 20 \\[4pt]
& x_i \in \{0,1\}, \qquad \forall i\in I .
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(I\) | Index set of courses | `file_0_view_0` | `course_id` (filtered where `discipline = 'Operations Research'`) |
| \(c_i\) | Credits | `file_0_view_0` | `credits` |
| \(p_i\) | Interest points | `file_0_view_0` | `interest_points` |
| \(x_i\) | Decision variable | – | – |

*All parameter values are taken directly from the specified columns of the filtered view `file_0_view_0`. No additional data or assumptions are introduced.*