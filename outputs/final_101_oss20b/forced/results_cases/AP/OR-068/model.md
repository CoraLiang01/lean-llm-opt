**Mathematical Model**

Let  

- \(I=\{MA,MB,MC,MD,ME,MF\}\) (Managers)  
- \(J=\{P1,P2,P3,P4,P5,P6\}\) (Projects)  

Parameters  

- \(c_{ij}\) : cost of assigning manager \(i\) to project \(j\)  

Decision variables  

- \(x_{ij}\in\{0,1\}\) : 1 if manager \(i\) is assigned to project \(j\), 0 otherwise  

Objective  

\[
\min \;\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

1. **Manager assignment**  
\[
\sum_{j\in J} x_{ij}=1 \quad \forall i\in I
\]

2. **Project assignment**  
\[
\sum_{i\in I} x_{ij}=1 \quad \forall j\in J
\]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(c_{ij}\) | Table **file_0_view_0** | For each row where `Unnamed: 0 = i` and each column `Pj`, set \(c_{ij}\) to the numeric value in that cell. |

*Table `file_0_view_0` columns:* `Unnamed: 0`, `P1`, `P2`, `P3`, `P4`, `P5`, `P6`.  
*Rows:* 6 managers (`MA`–`MF`).  
*No other data sources are used.*