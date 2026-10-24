**Mathematical Model**

Let  

* \(I=\{i\mid i\text{ is a course in the Operations Research discipline}\}\)  
* \(x_i\in\{0,1\}\) – binary variable, 1 if course \(i\) is selected, 0 otherwise  
* \(c_i\) – credits of course \(i\)  
* \(p_i\) – interest‑points of course \(i\)

\[
\begin{aligned}
\max\;&\sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t. } &\sum_{i\in I} c_i\,x_i \;\le\; 20 \\[4pt]
&x_i \in \{0,1\}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column | Filter |
|-----------|--------|----------|--------|--------|
| \(c_i\) | Credits | `file_0_view_0` | `credits` | `discipline = "Operations Research"` |
| \(p_i\) | Interest points | `file_0_view_0` | `interest_points` | `discipline = "Operations Research"` |
| \(I\) | Course IDs | `file_0_view_0` | `course_id` | `discipline = "Operations Research"` |

All values for \(c_i\) and \(p_i\) are taken directly from the rows of `courses_42.csv` that satisfy the discipline filter. No additional data or assumptions are introduced.