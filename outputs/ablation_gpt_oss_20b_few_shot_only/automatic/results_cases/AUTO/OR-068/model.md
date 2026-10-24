**Mathematical Model**

Let  

- \(M=\{MA,MB,MC,MD,ME,MF\}\) (Managers)  
- \(P=\{P1,P2,P3,P4,P5,P6\}\) (Projects)  

Parameters  

- \(c_{mp}\) = cost of assigning manager \(m\in M\) to project \(p\in P\).

Decision variables  

- \(x_{mp}\in\{0,1\}\) = 1 if manager \(m\) is assigned to project \(p\), 0 otherwise.

\[
\begin{aligned}
\min\;&\sum_{m\in M}\sum_{p\in P} c_{mp}\,x_{mp} \\[4pt]
\text{s.t.}\;&\sum_{p\in P} x_{mp}=1 &&\forall m\in M \\[2pt]
&\sum_{m\in M} x_{mp}=1 &&\forall p\in P \\[2pt]
&x_{mp}\in\{0,1\} &&\forall m\in M,\;p\in P
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(c_{mp}\) | `Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv` | For each manager row label \(m\) and project column label \(p\), \(c_{mp}\) equals the numeric value in that cell. The first column (`Unnamed: 0`) contains the manager identifiers; columns `P1`–`P6` contain the project identifiers. |

No other data or constraints are required.