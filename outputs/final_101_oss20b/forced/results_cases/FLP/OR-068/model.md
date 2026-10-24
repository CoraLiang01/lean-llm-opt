**Mathematical Model**

Let  

- \(I=\{MA,MB,MC,MD,ME,MF\}\) (Managers)  
- \(J=\{P1,P2,P3,P4,P5,P6\}\) (Projects)  

Parameters  

- \(c_{ij}\) : cost of assigning manager \(i\) to project \(j\) (from the CSV).  

Decision variables  

- \(x_{ij}\in\{0,1\}\) : 1 if manager \(i\) is assigned to project \(j\), 0 otherwise.  

Objective  

\[
\min \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

1. **Manager assignment** \(\displaystyle \sum_{j\in J} x_{ij}=1,\quad \forall i\in I\)  
2. **Project assignment** \(\displaystyle \sum_{i\in I} x_{ij}=1,\quad \forall j\in J\)  

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(c_{ij}\) | `file_0_view_0` | For each row where `Unnamed: 0 = i` and each column `Pj = j`, set \(c_{ij}\) to the numeric value in that cell. |

The table `file_0_view_0` contains the full cost matrix; no additional data is required.