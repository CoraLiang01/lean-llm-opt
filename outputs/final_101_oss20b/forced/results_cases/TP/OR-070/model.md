**Mathematical Model**

Let  

- \(M=\{1,\dots ,7\}\) be the set of managers (index \(m\)).  
- \(P=\{1,\dots ,7\}\) be the set of projects (index \(p\)).  
- \(c_{mp}\) be the cost for manager \(m\) to complete project \(p\).  

Decision variable  

\[
x_{mp}\in\{0,1\}\quad\forall m\in M,\;p\in P
\]

(1 if manager \(m\) is assigned to project \(p\), 0 otherwise).

**Objective**

\[
\min \sum_{m\in M}\sum_{p\in P} c_{mp}\,x_{mp}
\]

**Assignment constraints**

\[
\sum_{p\in P} x_{mp}=1 \quad\forall m\in M \qquad\text{(each manager gets one project)}
\]

\[
\sum_{m\in M} x_{mp}=1 \quad\forall p\in P \qquad\text{(each project gets one manager)}
\]

---

**Data Mapping Rules**

| Element | Source | Mapping |
|---------|--------|---------|
| Index set \(M\) | `Manager` column of table `file_0_view_0` | Each distinct value → element of \(M\) |
| Index set \(P\) | `Project 1 Cost` … `Project 7 Cost` columns of table `file_0_view_0` | Each column name → element of \(P\) |
| Parameter \(c_{mp}\) | Table `file_0_view_0` rows | For row with `Manager = Manager i` and column `Project j Cost`, set \(c_{ij}\) to the numeric value in that cell |
| Variable \(x_{mp}\) | Decision variable | Binary domain, defined for all \(m\in M, p\in P\) |
| Objective | Sum of \(c_{mp}x_{mp}\) | Minimize |
| Constraints | Assignment constraints | As listed above |

All numeric values are taken verbatim from the CSV; no aggregation or transformation is performed. The model uses exactly the 7 managers and 7 projects present in the source table.