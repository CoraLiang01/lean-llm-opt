## Mathematical Model

**Index sets**  
\(M=\{MA,MB,MC,MD,ME,MF\}\) (managers)  
\(P=\{P1,P2,P3,P4,P5,P6\}\) (projects)

**Parameters**  
\(c_{mp}\) – assignment cost of manager \(m\) to project \(p\)

**Decision variables**  
\(x_{mp}\in\{0,1\}\) (1 if manager \(m\) is assigned to project \(p\), 0 otherwise)

**Objective**  
\[
\min \;\sum_{m\in M}\sum_{p\in P} c_{mp}\,x_{mp}
\]

**Constraints**

1. **Manager assignment** (each manager gets exactly one project)  
\[
\sum_{p\in P} x_{mp}=1 \qquad \forall m\in M
\]

2. **Project assignment** (each project gets exactly one manager)  
\[
\sum_{m\in M} x_{mp}=1 \qquad \forall p\in P
\]

---

## Data Mapping

| Table ID | Column | Mapping |
|----------|--------|---------|
| `file_0_view_0` | `Unnamed: 0` | Manager identifier \(m\) |
| `file_0_view_0` | `P1`–`P6` | Project identifier \(p\) |
| `file_0_view_0` | `Pj` | Cost value \(c_{mp}\) where row `Unnamed: 0` = \(m\) and column = \(p\) |

All parameters \(c_{mp}\) are taken directly from the corresponding cell in the CSV; no additional processing or enumeration of numeric values is required. The variable domain is binary as specified.