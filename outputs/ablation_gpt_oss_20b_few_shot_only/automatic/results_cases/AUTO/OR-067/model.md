**Mathematical Model**

Let  

- \(M=\{MA,MB,MC\}\) be the set of managers,  
- \(P=\{P1,P2,P3\}\) be the set of projects,  
- \(c_{mp}\) be the cost of assigning manager \(m\) to project \(p\),  
- \(x_{mp}\in\{0,1\}\) be a binary decision variable that equals 1 if manager \(m\) is assigned to project \(p\).

\[
\begin{aligned}
\min\;&\sum_{m\in M}\sum_{p\in P} c_{mp}\,x_{mp} \\[4pt]
\text{s.t. }&\sum_{p\in P} x_{mp}=1 &&\forall m\in M \quad\text{(each manager gets one project)}\\
&\sum_{m\in M} x_{mp}=1 &&\forall p\in P \quad\text{(each project gets one manager)}\\
&x_{mp}\in\{0,1\} &&\forall m\in M,\;p\in P
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(c_{mp}\) | `Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv` | For each manager \(m\) (row label in column **Unnamed: 0**) and each project \(p\) (column header **P1**, **P2**, **P3**), set \(c_{mp}\) to the numeric value in that cell. |

No other data or constraints are required.