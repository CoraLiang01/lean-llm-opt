**Mathematical Model**

Let  

- \(M=\{m_1,\dots ,m_7\}\) (Managers)  
- \(P=\{p_1,\dots ,p_7\}\) (Projects)  

Parameters  

- \(c_{mp}\) = cost for manager \(m\) to complete project \(p\) (from the CSV).  

Decision variables  

- \(x_{mp}\in\{0,1\}\) = 1 if manager \(m\) is assigned to project \(p\), 0 otherwise.  

**Objective**

\[
\min \;\sum_{m\in M}\sum_{p\in P} c_{mp}\,x_{mp}
\]

**Constraints**

1. **Manager assignment**  
   \[
   \sum_{p\in P} x_{mp}=1 \quad \forall m\in M
   \]

2. **Project assignment**  
   \[
   \sum_{m\in M} x_{mp}=1 \quad \forall p\in P
   \]

3. **Binary variables**  
   \[
   x_{mp}\in\{0,1\}\quad \forall m\in M,\;p\in P
   \]

---

**Data Mapping**

| Table ID | Column | Index Set / Parameter |
|----------|--------|-----------------------|
| file_0_view_0 | Manager | \(M\) (maps each distinct Manager value to an element of \(M\)) |
| file_0_view_0 | Project 1 Cost | \(c_{m,p_1}\) |
| file_0_view_0 | Project 2 Cost | \(c_{m,p_2}\) |
| file_0_view_0 | Project 3 Cost | \(c_{m,p_3}\) |
| file_0_view_0 | Project 4 Cost | \(c_{m,p_4}\) |
| file_0_view_0 | Project 5 Cost | \(c_{m,p_5}\) |
| file_0_view_0 | Project 6 Cost | \(c_{m,p_6}\) |
| file_0_view_0 | Project 7 Cost | \(c_{m,p_7}\) |

*Each parameter \(c_{mp}\) is obtained by selecting the row where **Manager** = \(m\) and the column corresponding to the project cost (e.g., “Project 3 Cost” for \(p_3\)).*