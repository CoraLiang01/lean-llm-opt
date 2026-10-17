**Mathematical Model**

Let  

* \(I=\{M1,\dots ,M12\}\) – set of machines (index \(i\))  
* \(J=\{A,\dots ,L\}\)   – set of tasks (index \(j\))

Decision variables  

\[
x_{ij}\in\{0,1\}\qquad \forall i\in I,\; j\in J
\]

*\(x_{ij}=1\) if machine \(i\) is assigned to task \(j\), otherwise \(0\).*

Parameters  

\[
c_{ij}\ge 0 \qquad \forall i\in I,\; j\in J
\]

*\(c_{ij}\) is the machining cost of assigning machine \(i\) to task \(j\).*

---

**Objective**

\[
\min \;\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

---

**Constraints**

1. **One task per machine**

\[
\sum_{j\in J} x_{ij}=1 \qquad \forall i\in I
\]

2. **One machine per task**

\[
\sum_{i\in I} x_{ij}=1 \qquad \forall j\in J
\]

3. **Binary decision variables**

\[
x_{ij}\in\{0,1\}\qquad \forall i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source Table | Key Columns | Value Column |
|-----------|--------------|-------------|--------------|
| \(c_{ij}\) | `file_0_view_0` | `Machine` (index \(i\)) | `A`–`L` (index \(j\)) |

*The table `file_0_view_0` contains 12 rows (machines) and 12 columns (tasks). Each cell value is the cost \(c_{ij}\). No additional transformations are applied.*