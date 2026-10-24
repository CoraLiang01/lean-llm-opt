**Mathematical Model**

Let  

* \(I=\{\,\text{Manager }1,\dots ,\text{Manager }7\,\}\) – set of managers  
* \(J=\{\,\text{Project }1,\dots ,\text{Project }7\,\}\) – set of projects  

Decision variables  

\[
x_{ij}\in\{0,1\}\quad\forall i\in I,\;j\in J
\]

*\(x_{ij}=1\) if manager \(i\) is assigned to project \(j\); otherwise \(0\).*

Parameters  

\[
c_{ij}\ge 0 \quad\forall i\in I,\;j\in J
\]

*\(c_{ij}\) is the cost for manager \(i\) to complete project \(j\).*

**Objective**

\[
\min \;\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Assignment constraints**

\[
\sum_{j\in J} x_{ij}=1 \quad\forall i\in I \qquad\text{(each manager gets exactly one project)}
\]

\[
\sum_{i\in I} x_{ij}=1 \quad\forall j\in J \qquad\text{(each project gets exactly one manager)}
\]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(c_{ij}\) | Table **file_0_view_0** (column “Manager”, columns “Project 1 Cost” … “Project 7 Cost”) | For each row where *Manager* = \(i\) and column “Project \(j\) Cost” = value, set \(c_{ij}\) to that value. |

*No other parameters or data are required.*