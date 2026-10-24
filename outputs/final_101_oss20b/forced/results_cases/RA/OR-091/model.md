**Mathematical Model**

Let  

- \(I\) be the set of Operations Research course identifiers.  
- For each \(i\in I\)  
  - \(c_i\) = credits of course \(i\)  
  - \(p_i\) = interest‑points of course \(i\)  

Decision variables  

\[
x_i \in \{0,1\}\qquad \forall i\in I
\]

Objective  

\[
\max \;\sum_{i\in I} p_i\,x_i
\]

Capacity constraint  

\[
\sum_{i\in I} c_i\,x_i \;\le\; 20
\]

---

**Data Mapping**

| Parameter | Source | Column |
|-----------|--------|--------|
| \(c_i\) | `file_0_view_0` | `credits` |
| \(p_i\) | `file_0_view_0` | `interest_points` |
| Index \(i\) | `file_0_view_0` | `course_id` |

All values for the parameters are taken directly from the rows of `file_0_view_0` that satisfy the discipline filter “Operations Research”.