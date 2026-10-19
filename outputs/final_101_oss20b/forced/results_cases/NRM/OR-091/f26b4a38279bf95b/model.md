**Abstract Optimization Model**

**Index Set**  
\(I\) – set of all courses belonging to the *Operations Research* discipline.  

**Parameters**  
\[
\begin{aligned}
c_i &\;=\; \text{credits of course } i, && i\in I,\\
p_i &\;=\; \text{interest points of course } i, && i\in I.
\end{aligned}
\]

**Decision Variables**  
\[
x_i \;\in\; \{0,1\}, \qquad i\in I
\]
\(x_i=1\) if course \(i\) is selected, \(x_i=0\) otherwise.

**Objective Function**  
\[
\max \;\sum_{i\in I} p_i\,x_i
\]

**Constraints**  

1. **Credit limit**  
\[
\sum_{i\in I} c_i\,x_i \;\le\; 20
\]

2. **Binary decision variables**  
\[
x_i \;\in\; \{0,1\}, \qquad \forall i\in I
\]

---

### Data Mapping

| Source | Table ID | Columns Used |
|--------|----------|--------------|
| Course catalog | `file_0_view_0` | `course_id`, `credits`, `interest_points` |

*The discipline filter `discipline = "Operations Research"` is applied to construct the index set \(I\). No record values or counts are reproduced.*