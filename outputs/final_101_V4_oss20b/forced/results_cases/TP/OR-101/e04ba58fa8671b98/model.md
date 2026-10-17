**Mathematical Model**

Let  

* \(I=\{A,B,C,D,E,F,G,H,I,J\}\) – set of heavy‑machinery devices.  
* \(J=\{P1,P2,\dots ,P111\}\) – set of products.  

Decision variables  

\[
x_{ij}\;\ge 0 \qquad \forall i\in I,\; j\in J
\]

*\(x_{ij}\)* = quantity of product *j* produced on device *i* (continuous).

Parameters  

* \(\text{profit}_j\) – unit profit of product *j*.  
* \(\text{time}_{ij}\) – processing time (hours) required for one unit of product *j* on device *i*.  
* \(\text{cap}_i\) – monthly operating capacity (hours) of device *i*.

Objective  

\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{profit}_j\,x_{ij}
\]

Capacity constraints  

\[
\sum_{j\in J}\text{time}_{ij}\,x_{ij}\;\le\;\text{cap}_i
\qquad \forall i\in I
\]

Non‑negativity  

\[
x_{ij}\;\ge\;0 \qquad \forall i\in I,\; j\in J
\]

---

**Data‑Mapping Rules**

| Symbol | Source Table | Key Column(s) | Value Column(s) | Mapping |
|--------|--------------|---------------|-----------------|---------|
| \(I\) | `file_0_view_0` | `Device` | – | All distinct `Device` values (rows). |
| \(J\) | `file_0_view_0` | – | – | All distinct column headers `P1`–`P111`. |
| \(\text{time}_{ij}\) | `file_0_view_0` | `Device` (row), column header `Pj` | – | Matrix entry at row `Device=i`, column `Pj`. |
| \(\text{cap}_i\) | `file_1_view_0` | `Device` | `Monthly_Capacity` | Value where `Device=i`. |
| \(\text{profit}_j\) | `file_2_view_0` | `Product` | `Unit_Profit` | Value where `Product=j`. |

*All identifiers (`Device`, `Product`) are taken verbatim from the source tables; no aggregation or renaming is performed.*