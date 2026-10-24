**Mathematical Model**

Let  

* \(S=\{S_1,\dots ,S_{10}\}\) – source locations  
* \(D=\{D_1,\dots ,D_{20}\}\) – destination locations  

Parameters (all taken directly from the supplied tables)

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(c_{ij}\) | unit transportation cost from \(S_i\) to \(D_j\) | `file_0_view_0` (column `D_j`, row `source_id=S_i`) |
| \(s_i\) | available supply at source \(S_i\) | `file_2_view_0` (column `supply_units`) |
| \(d_j\) | demand at destination \(D_j\) | `file_1_view_0` (column `demand_units`) |

Decision variables  

* \(q_{ij}\ge 0\) – quantity of cargo shipped from \(S_i\) to \(D_j\) (continuous, in units)  
* \(t_{ij}\in \mathbb{Z}_{\ge 0}\) – number of trucks dispatched on route \((i,j)\)

Capacity of a truck: 10 units.  
Partial loading is allowed, so the quantity shipped on a route cannot exceed the
capacity of the trucks used on that route:

\[
q_{ij}\;\le\;10\,t_{ij}\qquad \forall i\in S,\;j\in D
\]

**Objective**

\[
\min \;\sum_{i\in S}\sum_{j\in D} c_{ij}\,q_{ij}
\]

**Constraints**

1. **Demand satisfaction** – each destination receives exactly its demand

\[
\sum_{i\in S} q_{ij}=d_j \qquad \forall j\in D
\]

2. **Supply limitation** – each source does not ship more than its available supply

\[
\sum_{j\in D} q_{ij}\le s_i \qquad \forall i\in S
\]

3. **Truck capacity linkage** – already listed above

4. **Variable domains**

\[
q_{ij}\ge 0,\qquad t_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in S,\;j\in D
\]

---

### Data Mapping

| Table ID | Columns | Mapping |
|----------|---------|---------|
| `file_0_view_0` | `source_id`, `D1`–`D20` | `c_{ij}` = value in column `D_j` for row with `source_id = S_i` |
| `file_1_view_0` | `destination_id`, `demand_units` | `d_j` = `demand_units` for row with `destination_id = D_j` |
| `file_2_view_0` | `source_id`, `supply_units` | `s_i` = `supply_units` for row with `source_id = S_i` |

All indices \(i\) and \(j\) run over the full sets of sources and destinations
present in the respective tables; no additional filtering is applied.