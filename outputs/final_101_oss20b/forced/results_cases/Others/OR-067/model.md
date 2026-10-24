## Mathematical Model

**Sets**  
\(M=\{MA,\,MB,\,MC\}\) (managers)  
\(P=\{P1,\,P2,\,P3\}\) (projects)

**Parameters**  
\(c_{mp}\) – cost for manager \(m\) to manage project \(p\)  
\(c_{mp}\) is taken from *file_0_view_0* (see Data Mapping)

**Decision Variables**  
\(x_{mp}\in\{0,1\}\) =1 if manager \(m\) is assigned to project \(p\), 0 otherwise

**Objective**  
\[
\min Z=\sum_{m\in M}\sum_{p\in P} c_{mp}\,x_{mp}
\]

**Constraints**

1. **Manager assignment** – each manager is assigned to exactly one project  
\[
\sum_{p\in P} x_{mp}=1 \qquad \forall m\in M
\]

2. **Project assignment** – each project is managed by exactly one manager  
\[
\sum_{m\in M} x_{mp}=1 \qquad \forall p\in P
\]

3. **Binary variables**  
\[
x_{mp}\in\{0,1\}\qquad \forall m\in M,\;p\in P
\]

---

## Data Mapping

| Element | Source | Mapping Details |
|---------|--------|-----------------|
| **Index set \(M\)** | *file_0_view_0* – column **Unnamed: 0** | Values: `MA`, `MB`, `MC` |
| **Index set \(P\)** | *file_0_view_0* – columns **P1**, **P2**, **P3** | Values: `P1`, `P2`, `P3` |
| **Parameter \(c_{mp}\)** | *file_0_view_0* – cell at row where **Unnamed: 0**=`m` and column=`p` | Numeric cost (e.g., `c_{MA,P1}=3000`) |
| **Variable \(x_{mp}\)** | Decision variable | Binary, defined for all \((m,p)\in M\times P\) |
| **Objective** | Sum of \(c_{mp}x_{mp}\) | Minimize |
| **Constraints** | Assignment constraints | Equality constraints as listed above |

All parameters, variables, and constraints are derived directly from the data in *file_0_view_0*; no additional data or assumptions are introduced.