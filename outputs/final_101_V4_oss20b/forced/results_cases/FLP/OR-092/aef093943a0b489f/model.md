**Mathematical Model**

Let  

- \(I=\{S1,\dots ,S10\}\) : set of source locations  
- \(J=\{D1,\dots ,D20\}\) : set of destination locations  

Parameters  

- \(c_{ij}\) : unit transportation cost from source \(i\) to destination \(j\)  
- \(s_i\)  : available supply units at source \(i\)  
- \(d_j\)  : required demand units at destination \(j\)

Decision variables  

- \(x_{ij}\ge 0\) : units shipped from \(i\) to \(j\) (continuous)  
- \(t_{ij}\in \mathbb{Z}_{+}\) : number of whole trucks dispatched from \(i\) to \(j\)

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } &\sum_{i\in I} x_{ij}= d_j , &&\forall j\in J \quad\text{(meet demand)}\\[4pt]
&\sum_{j\in J} x_{ij}\le s_i , &&\forall i\in I \quad\text{(respect supply)}\\[4pt]
&x_{ij}\le 10\,t_{ij} , &&\forall i\in I,\; j\in J \quad\text{(truck capacity)}\\[4pt]
&x_{ij}\ge 0 , &&\forall i\in I,\; j\in J\\[4pt]
&t_{ij}\in \mathbb{Z}_{+} , &&\forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Row Key | Column Key | Notes |
|-----------|--------------|---------|------------|-------|
| \(c_{ij}\) | `file_0_view_0` | `source_id` | `D1`–`D20` | Matrix of unit costs |
| \(s_i\) | `file_2_view_0` | `source_id` | `supply_units` | Supply at each source |
| \(d_j\) | `file_1_view_0` | `destination_id` | `demand_units` | Demand at each destination |

Indices \(I\) and \(J\) are taken directly from the distinct values of `source_id` and `destination_id` in the respective tables. No additional limits or constants are introduced beyond those specified.