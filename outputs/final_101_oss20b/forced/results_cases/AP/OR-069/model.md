**Mathematical Model**

Let  

* \(M\) – set of managers (rows of the CSV)  
* \(P\) – set of projects (columns of the CSV, excluding the “Manager” column)  

Parameters  

* \(c_{mp}\) – assignment cost for manager \(m\in M\) to project \(p\in P\)  

Decision variables  

* \(x_{mp}\in\{0,1\}\) – 1 if manager \(m\) is assigned to project \(p\), 0 otherwise  

Objective  

\[
\min \;\sum_{m\in M}\sum_{p\in P} c_{mp}\,x_{mp}
\]

Constraints  

1. **Manager assignment** – each manager gets exactly one project  

\[
\sum_{p\in P} x_{mp}=1 \quad \forall m\in M
\]

2. **Project assignment** – each project gets exactly one manager  

\[
\sum_{m\in M} x_{mp}=1 \quad \forall p\in P
\]

3. **Binary variables**  

\[
x_{mp}\in\{0,1\}\quad \forall m\in M,\;p\in P
\]

---

**Data Mapping**

| Table | ID | Column | Mapping |
|-------|----|--------|---------|
| manager_project_costs.csv | `file_0_view_0` | `Manager` | `m` (manager index) |
| | | `Project 1 Cost` | `p = 1` |
| | | `Project 2 Cost` | `p = 2` |
| | | … | … |
| | | `Project 11 Cost` | `p = 11` |

*For each row \(r\) and column \(c\) (excluding the Manager column):*  

\[
c_{\,\text{Manager}[r],\,p(c)} = \text{value in cell }(r,c)
\]

where \(p(c)\) is the integer extracted from the column name “Project \(p\) Cost”.  

The sets \(M\) and \(P\) are obtained from the distinct values in the `Manager` column and the project numbers 1–11, respectively. All costs are treated as real (integer) parameters.