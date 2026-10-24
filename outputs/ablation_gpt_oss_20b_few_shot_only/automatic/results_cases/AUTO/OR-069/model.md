**Mathematical model**

Let  

- \(M=\{m\mid m\text{ is a manager in the CSV file}\}\)  
- \(P=\{p\mid p\text{ is a project in the CSV file}\}\)

Parameters  

\[
c_{m,p}\in\mathbb{R}\qquad\forall m\in M,\;p\in P
\]

Decision variables  

\[
x_{m,p}\in\{0,1\}\qquad\forall m\in M,\;p\in P
\]

Objective  

\[
\min \;\sum_{m\in M}\sum_{p\in P} c_{m,p}\,x_{m,p}
\]

Constraints  

\[
\sum_{p\in P} x_{m,p}=1 \qquad\forall m\in M \quad\text{(each manager assigned once)}
\]

\[
\sum_{m\in M} x_{m,p}=1 \qquad\forall p\in P \quad\text{(each project assigned once)}
\]

---

**Data Mapping**

| Parameter | Source | Mapping rule |
|-----------|--------|--------------|
| \(c_{m,p}\) | `Test_Dataset/Large-scale-or/AP_testing/AP4/manager_project_costs.csv` | For each row with header **Manager** value `m` and each column with header **Project k Cost** value `p`, set \(c_{m,p}\) to the numeric entry in that cell. |
| \(x_{m,p}\) | Decision variable | Binary domain: \(x_{m,p}\in\{0,1\}\). |

*Indices \(M\) and \(P\) are taken directly from the distinct Manager and Project identifiers present in the CSV file.*