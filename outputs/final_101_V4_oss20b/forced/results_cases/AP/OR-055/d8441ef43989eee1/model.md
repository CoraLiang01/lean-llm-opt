**Mathematical Model**

Let  

\[
I=\{\,i \mid i=\text{DisplayID from file\_0\_view\_0}\,\}, \qquad
J=\{\,j \mid j=\text{ProductName from file\_1\_view\_0}\,\}.
\]

Parameters  

\[
\begin{aligned}
\text{capacity}_i &\;=\; \text{Capacity of display } i \quad (i\in I),\\
\text{value}_j     &\;=\; \text{Value of product } j \quad (j\in J),\\
\text{weight}_j    &\;=\; \text{Weight of product } j \quad (j\in J).
\end{aligned}
\]

Decision variables  

\[
x_{ij}\;\in\;\mathbb{Z}_{+}\quad (i\in I,\;j\in J)
\]

(number of units of product \(j\) placed in display \(i\)).

---

**Objective**

\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{value}_j\,x_{ij}
\]

---

**Capacity Constraints**

\[
\sum_{j\in J}\text{weight}_j\,x_{ij}\;\le\;\text{capacity}_i
\qquad \forall\, i\in I
\]

---

**Variable Domain**

\[
x_{ij}\;\ge\;0,\; x_{ij}\in\mathbb{Z}\qquad \forall\, i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| `capacity_i` | `file_0_view_0` | `Capacity` |
| `value_j`     | `file_1_view_0` | `Value` |
| `weight_j`    | `file_1_view_0` | `Weight` |

| Index Set | Source Table | Source Column |
|-----------|--------------|---------------|
| `I` | `file_0_view_0` | `DisplayID` |
| `J` | `file_1_view_0` | `ProductName` |

---