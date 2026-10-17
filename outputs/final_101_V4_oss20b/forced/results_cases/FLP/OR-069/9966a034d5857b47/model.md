**Mathematical Model**

Let  

- \(M\) be the set of managers (rows of the CSV).  
- \(P\) be the set of projects (columns of the CSV).  
- \(c_{ij}\) be the assignment cost for manager \(i\) to project \(j\).  
- \(x_{ij}\in\{0,1\}\) be a binary variable that equals 1 if manager \(i\) is assigned to project \(j\), 0 otherwise.

\[
\begin{aligned}
\min\;&\sum_{i\in M}\sum_{j\in P} c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{j\in P} x_{ij}=1 &&\forall i\in M \quad\text{(each manager gets one project)}\\
            &\sum_{i\in M} x_{ij}=1 &&\forall j\in P \quad\text{(each project gets one manager)}\\
            &x_{ij}\in\{0,1\} &&\forall i\in M,\; j\in P
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Mapping Rule |
|--------|--------|--------------|
| \(M\) | `file_0_view_0` | All distinct values in column **Manager** |
| \(P\) | `file_0_view_0` | All distinct values in columns **Project 1 Cost**, **Project 2 Cost**, …, **Project 11 Cost** |
| \(c_{ij}\) | `file_0_view_0` | For a given manager row \(i\) and project column \(j\), take the numeric value in the cell labeled “Project \(j\) Cost”. |

No numerical values are listed; the model references the CSV through the mapping rules above.