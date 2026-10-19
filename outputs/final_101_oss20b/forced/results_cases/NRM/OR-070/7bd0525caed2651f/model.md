##### Abstract Model

**Index Sets**  
\(M\) – set of managers (indexed by \(m\)).  
\(P\) – set of projects (indexed by \(p\)).

**Parameters**  
\(c_{mp}\) – cost incurred by manager \(m\) to complete project \(p\).

**Decision Variables**  
\(x_{mp}\) – binary variable,  
\[
x_{mp}= \begin{cases}
1 & \text{if manager } m \text{ is assigned to project } p,\\
0 & \text{otherwise.}
\end{cases}
\]

**Objective Function**  
\[
\min \quad \sum_{m \in M}\sum_{p \in P} c_{mp}\,x_{mp}
\]

**Constraints**

1. **Project Assignment** – each project is assigned to exactly one manager  
\[
\sum_{m \in M} x_{mp} = 1, \quad \forall p \in P
\]

2. **Manager Capacity** – each manager is assigned to at most one project  
\[
\sum_{p \in P} x_{mp} \le 1, \quad \forall m \in M
\]

3. **Variable Domain** – binary decision variables  
\[
x_{mp} \in \{0,1\}, \quad \forall m \in M,\; \forall p \in P
\]

---

##### Data Mapping

| Source | Table ID | Column Names |
|--------|----------|--------------|
| Manager‑Project Cost Matrix | `file_0_view_0` | `Manager`, `Project 1 Cost`, `Project 2 Cost`, `Project 3 Cost`, `Project 4 Cost`, `Project 5 Cost`, `Project 6 Cost`, `Project 7 Cost` |

The parameter \(c_{mp}\) is populated from the corresponding cost column for each manager in the table `file_0_view_0`.