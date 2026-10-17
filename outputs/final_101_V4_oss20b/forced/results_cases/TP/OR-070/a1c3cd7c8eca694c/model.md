**Mathematical Model**

Let  

* \(M=\{m_1,\dots ,m_7\}\) – set of managers (rows of *manager_project_costs.csv*)  
* \(P=\{p_1,\dots ,p_7\}\) – set of projects (columns “Project 1 Cost” … “Project 7 Cost”)  

Parameters  

* \(c_{mp}\) – cost for manager \(m\) to complete project \(p\) (taken from the corresponding cell in the CSV).  

Decision variables  

* \(x_{mp}\in\{0,1\}\) – 1 if manager \(m\) is assigned to project \(p\), 0 otherwise.  

\[
\begin{aligned}
\min\;&\sum_{m\in M}\sum_{p\in P} c_{mp}\,x_{mp} \\[4pt]
\text{s.t.}\;&\sum_{p\in P} x_{mp}=1 &&\forall m\in M \quad\text{(each manager gets one project)}\\[2pt]
&\sum_{m\in M} x_{mp}=1 &&\forall p\in P \quad\text{(each project gets one manager)}\\[2pt]
&x_{mp}\in\{0,1\} &&\forall m\in M,\;p\in P
\end{aligned}
\]

---

**Data Mapping Rules**

| Data element | Source | Mapping |
|--------------|--------|---------|
| Index set \(M\) | `file_0_view_0` column **Manager** | Each distinct value → element of \(M\) |
| Index set \(P\) | `file_0_view_0` columns **Project 1 Cost** … **Project 7 Cost** | Column *Project k Cost* → element \(p_k\) of \(P\) |
| Parameter \(c_{mp}\) | `file_0_view_0` cell at row *Manager i*, column **Project k Cost** | Value → \(c_{m_i,p_k}\) |
| Variable \(x_{mp}\) | Decision variable | Binary domain, defined for all \((m,p)\in M\times P\) |
| Objective | Minimize | \(\sum_{m,p} c_{mp}x_{mp}\) |
| Constraints | Assignment constraints | As listed in the model above |

All identifiers and coefficients are taken verbatim from the CSV; no aggregation or transformation is performed. The model is ready for direct implementation in any MILP solver.