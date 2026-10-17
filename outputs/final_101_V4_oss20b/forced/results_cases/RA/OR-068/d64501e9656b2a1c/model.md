**Mathematical Model**

Let  

- \(M=\{MA,MB,MC,MD,ME,MF\}\) (index of managers)  
- \(P=\{P1,P2,P3,P4,P5,P6\}\) (index of projects)  

Decision variables  

\[
x_{mp}\in\{0,1\}\quad\forall m\in M,\;p\in P
\]

Parameter  

\[
c_{mp}\quad\text{(assignment cost of manager }m\text{ to project }p)
\]

Objective  

\[
\min \;\sum_{m\in M}\sum_{p\in P} c_{mp}\,x_{mp}
\]

Constraints  

1. **Manager assignment**  
\[
\sum_{p\in P} x_{mp}=1 \quad\forall m\in M
\]

2. **Project assignment**  
\[
\sum_{m\in M} x_{mp}=1 \quad\forall p\in P
\]

---

**Data Mapping**

| Parameter / Index | Source Table | Column(s) | Mapping |
|-------------------|--------------|-----------|---------|
| \(M\) | `file_0_view_0` | `Unnamed: 0` | Each distinct value in `Unnamed: 0` defines a manager \(m\). |
| \(P\) | `file_0_view_0` | `P1`–`P6` | Each column name `Pj` defines a project \(p\). |
| \(c_{mp}\) | `file_0_view_0` | `P1`–`P6` | For a given row with manager `Unnamed: 0 = m`, the cell in column `Pj` gives \(c_{mp}\). |

All other data are derived from these mappings. No additional parameters or constraints are introduced.