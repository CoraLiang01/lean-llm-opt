## Symbolic Mathematical Model

**Sets**

- $I$: set of candidate sensor sites (from sensor_sites.csv, column Center)
- $J$: set of monitoring zones (from monitoring_zones.csv, column Zone)
- $C_i$: opening cost for site $i \in I$ (from sensor_sites.csv, OpeningCost)
- $A_{ij}$: coverage indicator, $A_{ij} = 1$ if site $i$ covers zone $j$, $0$ otherwise (from sensor_sites.csv, CoveredDistricts)

**Parameters**

- $C_i$: installation cost for site $i \in I$
- $A_{ij}$: 
  $$
  A_{ij} = 
  \begin{cases}
    1 & \text{if zone } j \text{ is in CoveredDistricts of site } i \\
    0 & \text{otherwise}
  \end{cases}
  $$

**Decision Variables**

- $y_i \in \{0,1\}$, for $i \in I$: $y_i = 1$ if a sensor is installed at site $i$, $0$ otherwise

**Objective**

Minimize total installation cost:
$$
\min \sum_{i \in I} C_i y_i
$$

**Constraints**

Coverage: For each monitoring zone $j \in J$, at least one installed sensor must cover it:
$$
\sum_{i \in I} A_{ij} y_i \geq 1 \qquad \forall j \in J
$$

Binary restrictions:
$$
y_i \in \{0,1\} \qquad \forall i \in I
$$

---

## Data Mapping

- $I$ (sensor sites): sensor_sites.csv, column Center, table_id: file_0_view_0
- $J$ (monitoring zones): monitoring_zones.csv, column Zone, table_id: file_1_view_0
- $C_i$: sensor_sites.csv, column OpeningCost, table_id: file_0_view_0
- $A_{ij}$: $A_{ij} = 1$ if $j$ is in CoveredDistricts of $i$ (sensor_sites.csv, CoveredDistricts, table_id: file_0_view_0), $0$ otherwise

**Complete Model:**
$$
\begin{align*}
\min\ & \sum_{i \in I} C_i y_i \\
\text{s.t.}\quad & \sum_{i \in I} A_{ij} y_i \geq 1 \qquad \forall j \in J \\
& y_i \in \{0,1\} \qquad \forall i \in I
\end{align*}
$$

**All sets, parameters, and variables are mapped directly to the current CSV data as described above.**