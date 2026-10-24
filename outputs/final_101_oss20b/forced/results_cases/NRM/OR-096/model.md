**Symbolic Model**

\[
\begin{aligned}
\text{Sets: } & S=\{I,II\} \quad\text{(schools)}\\
& N=\{N01,\dots ,N31\}\quad\text{(neighborhoods)}\\
& R=\{W,NW\}\quad\text{(race: White, Non‑White)}\\[4pt]

\text{Parameters: } & \text{Cap}_{s}\quad\text{(school capacity)}\\
& \text{Pop}_{n,r}\quad\text{(population of race }r\text{ in neighborhood }n)\\
& \text{Dist}_{s,n}\quad\text{(distance from school }s\text{ to neighborhood }n)\\[4pt]

\text{Decision variables: } & x_{s,n,r}\in\mathbb{Z}_{+}\quad\text{(pupils of race }r\text{ from }n\text{ assigned to }s)\\[4pt]

\text{Objective: } & 
\min \sum_{s\in S}\sum_{n\in N}\text{Dist}_{s,n}\,
\bigl(x_{s,n,W}+x_{s,n,NW}\bigr)\\[6pt]

\text{Constraints: }\\
&\text{(1) School capacity}\qquad
\sum_{n\in N}\bigl(x_{s,n,W}+x_{s,n,NW}\bigr)\le \text{Cap}_{s},
\quad \forall s\in S\\[4pt]
&\text{(2) Neighborhood population}\qquad
\sum_{s\in S}x_{s,n,r}= \text{Pop}_{n,r},
\quad \forall n\in N,\; r\in R\\[4pt]
&\text{(3) Racial balance (60 % ± 10 %) }\qquad
0.5\,\sum_{n\in N}\bigl(x_{s,n,W}+x_{s,n,NW}\bigr)
\le \sum_{n\in N}x_{s,n,W}
\le 0.7\,\sum_{n\in N}\bigl(x_{s,n,W}+x_{s,n,NW}\bigr),
\quad \forall s\in S\\[4pt]
&\text{(4) Non‑negativity and integrality}\qquad
x_{s,n,r}\in\mathbb{Z}_{+},\quad \forall s\in S,\; n\in N,\; r\in R
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Columns |
|-----------|--------------|----------------|
| `Cap_s` | `file_0_view_0` | `School`, `Capacity` |
| `Pop_{n,W}` | `file_1_view_0` | `Neighborhood`, `Population_White` |
| `Pop_{n,NW}` | `file_1_view_0` | `Neighborhood`, `Population_NonWhite` |
| `Dist_{s,n}` | `file_2_view_0` | `School`, `N01`–`N31` (each column is a neighborhood) |

All parameters are taken directly from the corresponding columns of the specified tables; no numerical values are listed here.