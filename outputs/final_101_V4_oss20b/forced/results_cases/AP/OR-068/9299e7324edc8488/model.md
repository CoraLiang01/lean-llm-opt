**Mathematical Model**

Let  

- \(I=\{MA,MB,MC,MD,ME,MF\}\) (Managers)  
- \(J=\{P1,P2,P3,P4,P5,P6\}\) (Projects)  

Parameters  

- \(c_{ij}\) : cost of assigning manager \(i\) to project \(j\)  

Decision variables  

- \(x_{ij}\in\{0,1\}\) : 1 if manager \(i\) is assigned to project \(j\), 0 otherwise  

**Objective**

\[
\min \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Manager assignment** (each manager gets exactly one project)

\[
\sum_{j\in J} x_{ij}=1 \quad \forall i\in I
\]

2. **Project assignment** (each project gets exactly one manager)

\[
\sum_{i\in I} x_{ij}=1 \quad \forall j\in J
\]

3. **Binary variables**

\[
x_{ij}\in\{0,1\}\quad \forall i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(c_{ij}\) | Table **file_0_view_0** | For each manager row \(i\) (column *Unnamed: 0*) and each project column \(j\) (*P1*–*P6*), assign \(c_{ij}\) the numeric value in that cell. |

*No numerical values are enumerated in the model; they are retrieved from the specified CSV table.*