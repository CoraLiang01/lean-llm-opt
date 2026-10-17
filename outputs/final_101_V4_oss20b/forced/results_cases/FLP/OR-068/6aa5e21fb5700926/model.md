**Mathematical Model**

Let  

- \(I=\{MA,MB,MC,MD,ME,MF\}\) (Managers)  
- \(J=\{P1,P2,P3,P4,P5,P6\}\) (Projects)  

Parameters  

- \(c_{ij}\) – assignment cost of manager \(i\) to project \(j\) (from the CSV).  

Decision variables  

- \(x_{ij}\in\{0,1\}\) – equals 1 if manager \(i\) is assigned to project \(j\), 0 otherwise.  

Objective  

\[
\min \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

1. **Each manager is assigned to exactly one project**  
   \[
   \sum_{j\in J} x_{ij}=1,\qquad \forall i\in I
   \]

2. **Each project receives exactly one manager**  
   \[
   \sum_{i\in I} x_{ij}=1,\qquad \forall j\in J
   \]

3. **Binary assignment**  
   \[
   x_{ij}\in\{0,1\},\qquad \forall i\in I,\; j\in J
   \]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(c_{ij}\) | `file_0_view_0` | For each manager \(i\) (row where `Unnamed: 0` = \(i\)) and project \(j\) (column `Pj`), set \(c_{ij}\) to the cell value. |

*Table `file_0_view_0` columns:* `Unnamed: 0`, `P1`, `P2`, `P3`, `P4`, `P5`, `P6`.  
*Rows:* each row corresponds to a manager (`MA`–`MF`).  
*No other data sources are used.*