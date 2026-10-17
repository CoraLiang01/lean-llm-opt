**Mathematical Model**

Let  

- \(I=\{MA,\,MB,\,MC\}\) – set of managers  
- \(J=\{P1,\,P2,\,P3\}\) – set of projects  

Parameters  

- \(c_{ij}\) – cost for manager \(i\) to manage project \(j\)  

Decision variables  

- \(x_{ij}\in\{0,1\}\) – 1 if manager \(i\) is assigned to project \(j\), 0 otherwise  

Objective  

\[
\min \;\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

1. **One project per manager**  
   \[
   \sum_{j\in J} x_{ij}=1 \quad \forall\, i\in I
   \]

2. **One manager per project**  
   \[
   \sum_{i\in I} x_{ij}=1 \quad \forall\, j\in J
   \]

3. **Binary assignment**  
   \[
   x_{ij}\in\{0,1\}\quad \forall\, i\in I,\; j\in J
   \]

---

**Data Mapping Rules**

| Table ID | Source Column | Mapping to Model |
|----------|---------------|------------------|
| `file_0_view_0` | `Unnamed: 0` | Manager identifier → index \(i\in I\) |
| `file_0_view_0` | `P1` | Project identifier → index \(j\in J\) |
| `file_0_view_0` | `P2` | Project identifier → index \(j\in J\) |
| `file_0_view_0` | `P3` | Project identifier → index \(j\in J\) |
| `file_0_view_0` | cell \((i,j)\) | Cost value → parameter \(c_{ij}\) |

*The table `file_0_view_0` contains the full cost matrix; each row corresponds to a manager (identified by `Unnamed: 0`) and each column `P1`–`P3` corresponds to a project. The parameter \(c_{ij}\) is taken directly from the cell at the intersection of row \(i\) and column \(j\).*