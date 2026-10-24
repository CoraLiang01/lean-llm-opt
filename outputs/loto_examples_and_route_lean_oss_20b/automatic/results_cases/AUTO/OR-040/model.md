## Mathematical Model

**Index set**

\[
I=\{\,i\mid i \text{ is a ProductName in } \text{file\_1\_view\_0}\,\}
\]

**Parameters**

\[
v_i = \text{Value of product } i \quad (i\in I)
\]
\[
C   = \text{Capacity from file\_0\_view\_0}
\]

**Decision variables**

\[
x_i \in \mathbb{Z}_{+}\quad (i\in I)
\]

**Objective**

\[
\max \;\sum_{i\in I} v_i\,x_i
\]

**Constraints**

\[
\sum_{i\in I} x_i \;\le\; C
\]

---

## Data Mapping

| Parameter | Source Table | Source Column | Mapping |
|-----------|--------------|---------------|---------|
| \(C\) | `file_0_view_0` | `Capacity` | `C = Capacity` |
| \(v_i\) | `file_1_view_0` | `Value` | `v_i = Value` for each `ProductName` i |
| Index set \(I\) | `file_1_view_0` | `ProductName` | `I = {ProductName}` |

*The `Weight` column in `file_1_view_0` is not used in this model.*