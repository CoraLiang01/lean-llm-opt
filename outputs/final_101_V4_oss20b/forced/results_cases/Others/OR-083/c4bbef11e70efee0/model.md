## Mathematical Model

**Indices**  
\(W=\{1,2,\dots ,12\}\) – workers  
\(T=\{A,B,C,D,E,F,G,H,I,J\}\) – tasks  

**Parameters**  
\(t_{ij}\) – time (hours) worker \(i\) needs to complete task \(j\)  
(see Data Mapping below)

**Decision variables**  
\(x_{ij}\in\{0,1\}\) – 1 if worker \(i\) is assigned to task \(j\), 0 otherwise  

**Objective**  
\[
\min \;\sum_{i\in W}\sum_{j\in T} t_{ij}\,x_{ij}
\]

**Constraints**

1. **One worker per task**  
\[
\sum_{i\in W} x_{ij}=1 \qquad \forall j\in T
\]

2. **At most one task per worker**  
\[
\sum_{j\in T} x_{ij}\le 1 \qquad \forall i\in W
\]

3. **(Implicit) Exactly 10 workers used** – follows from (1) and (2).  
   (Optional explicit form)  
\[
\sum_{i\in W}\sum_{j\in T} x_{ij}=10
\]

---

## Data Mapping

| Table ID | Column | Mapping |
|----------|--------|---------|
| `file_0_view_0` | `Task Time Required` | Worker index \(i\) (values 1–12) |
| `file_0_view_0` | `A` | Task \(j=A\) |
| `file_0_view_0` | `B` | Task \(j=B\) |
| `file_0_view_0` | `C` | Task \(j=C\) |
| `file_0_view_0` | `D` | Task \(j=D\) |
| `file_0_view_0` | `E` | Task \(j=E\) |
| `file_0_view_0` | `F` | Task \(j=F\) |
| `file_0_view_0` | `G` | Task \(j=G\) |
| `file_0_view_0` | `H` | Task \(j=H\) |
| `file_0_view_0` | `I` | Task \(j=I\) |
| `file_0_view_0` | `J` | Task \(j=J\) |

The parameter \(t_{ij}\) is taken directly from the cell at row \(i\) and column \(j\) of this table. No additional data or transformations are required.