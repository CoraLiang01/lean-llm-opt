## Mathematical Model

**Sets**  
\(M=\{MA,MB,MC\}\) (managers)  
\(P=\{P1,P2,P3\}\) (projects)

**Parameters**  
\(c_{mp}\) – cost for manager \(m\) to manage project \(p\)  
\(c_{mp}\) is obtained from *file_0_view_0* (column “P1”, “P2”, “P3” for each row “Unnamed: 0”).

**Decision Variables**  
\(x_{mp}\in\{0,1\}\) =1 if manager \(m\) is assigned to project \(p\), 0 otherwise.

**Objective**  
\[
\min \;\sum_{m\in M}\sum_{p\in P} c_{mp}\,x_{mp}
\]

**Constraints**

1. **Manager assignment** (each manager to exactly one project)  
\[
\sum_{p\in P} x_{mp}=1 \quad \forall m\in M
\]

2. **Project assignment** (each project to exactly one manager)  
\[
\sum_{m\in M} x_{mp}=1 \quad \forall p\in P
\]

3. **Binary variables**  
\[
x_{mp}\in\{0,1\}\quad \forall m\in M,\;p\in P
\]

---

## Data Mapping

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(M\) | *file_0_view_0* | Set of distinct values in column **“Unnamed: 0”** |
| \(P\) | *file_0_view_0* | Set of column names **“P1”**, **“P2”**, **“P3”** |
| \(c_{mp}\) | *file_0_view_0* | For each row with manager \(m\) (value of **“Unnamed: 0”**) and each column \(p\) (one of **“P1”**, **“P2”**, **“P3”**), assign \(c_{mp}\) = numeric value in that cell |

No additional data or constraints are introduced beyond those explicitly stated in the query.