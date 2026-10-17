## Mathematical Model

**Index set**  
\(I=\{\text{ProductName}\}\) – all real‑estate areas listed in *products.csv*.

**Parameters**  
\[
\begin{aligned}
&\text{benefit}_{i} &&\text{(benefit coefficient for area }i\text{)} \\
&\text{capacity} &&\text{overall development capacity (from *capacity.csv*)}
\end{aligned}
\]

**Decision variables**  
\[
x_{i}\in \mathbb{Z}_{+}\qquad \forall i\in I
\]

**Objective**  
\[
\max \;\sum_{i\in I}\text{benefit}_{i}\,x_{i}
\]

**Constraints**  

1. **Capacity constraint**  
\[
\sum_{i\in I}x_{i}\;\le\;\text{capacity}
\]

2. **Non‑negativity & integrality**  
\[
x_{i}\ge 0,\; x_{i}\in\mathbb{Z}\qquad \forall i\in I
\]

---

## Data Mapping

| Table ID | Source File | Columns Used | Mapping |
|----------|-------------|--------------|---------|
| `file_0_view_0` | `capacity.csv` | `Capacity` | `capacity = Capacity` |
| `file_1_view_0` | `products.csv` | `ProductName`, `Value` | `benefit[ProductName] = Value` |

*All other columns in the source files are ignored for this model.*