**Mathematical Model**

Let  

* \(M=\{M_1,\dots ,M_{12}\}\) – set of machines  
* \(T=\{A,B,C,D,E,F,G,H,I,J,K,L\}\) – set of tasks  

Parameters  

* \(c_{ij}\) – machining cost of assigning machine \(i\in M\) to task \(j\in T\)  

Decision variables  

* \(x_{ij}\in\{0,1\}\) – equals 1 if machine \(i\) is assigned to task \(j\), 0 otherwise  

Objective  

\[
\min \;\sum_{i\in M}\sum_{j\in T} c_{ij}\,x_{ij}
\]

Constraints  

1. **Machine‑to‑task assignment**  
\[
\sum_{j\in T} x_{ij}=1 \quad \forall i\in M
\]

2. **Task‑to‑machine assignment**  
\[
\sum_{i\in M} x_{ij}=1 \quad \forall j\in T
\]

3. **Binary domain**  
\[
x_{ij}\in\{0,1\}\quad \forall i\in M,\; j\in T
\]

---

**Data Mapping Rules**

| Symbol | Source | Mapping |
|--------|--------|---------|
| \(M\) | `file_0_view_0` – column **Machine** | Each distinct value in the **Machine** column defines an element of \(M\). |
| \(T\) | `file_0_view_0` – columns **A**–**L** | Each column name **A** through **L** defines an element of \(T\). |
| \(c_{ij}\) | `file_0_view_0` – row where **Machine** = \(i\), column = \(j\) | The numeric entry in that cell is the cost \(c_{ij}\). |
| \(x_{ij}\) | Decision variable | Binary variable associated with pair \((i,j)\). |

No numerical values are listed here; the model references the data through the above mapping.