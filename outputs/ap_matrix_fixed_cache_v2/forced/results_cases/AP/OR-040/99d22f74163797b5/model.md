##### Decision Variables

Let $x_i$ be the integer number of development units chosen daily in area $i$, for each area $i$ in the set:
\[
\text{Areas} = \{
\text{Queens},\ 
\text{Brooklyn},\ 
\text{Manhattan},\ 
\text{Bronx},\ 
\text{Staten Island},\ 
\text{Harlem},\ 
\text{Upper East Side},\ 
\text{Lower Manhattan},\ 
\text{Midtown},\ 
\text{Long Island City},\ 
\text{Williamsburg},\ 
\text{Bushwick},\ 
\text{Flatbush},\ 
\text{Greenpoint},\ 
\text{Park Slope},\ 
\text{Astoria},\ 
\text{Jackson Heights},\ 
\text{Flushing},\ 
\text{Sunnyside},\ 
\text{Ditmars}
\}
\]

##### Parameters

- Benefit coefficients $v_i$ and weights $w_i$ for each area $i$:

\[
\begin{array}{lll}
\text{Area} & v_i & w_i \\
\hline
\text{Queens} & 443 & 104 \\
\text{Brooklyn} & 522 & 368 \\
\text{Manhattan} & 300 & 483 \\
\text{Bronx} & 767 & 165 \\
\text{Staten Island} & 300 & 105 \\
\text{Harlem} & 309 & 123 \\
\text{Upper East Side} & 598 & 131 \\
\text{Lower Manhattan} & 460 & 341 \\
\text{Midtown} & 318 & 258 \\
\text{Long Island City} & 126 & 469 \\
\text{Williamsburg} & 593 & 387 \\
\text{Bushwick} & 871 & 425 \\
\text{Flatbush} & 858 & 482 \\
\text{Greenpoint} & 321 & 495 \\
\text{Park Slope} & 275 & 305 \\
\text{Astoria} & 700 & 377 \\
\text{Jackson Heights} & 685 & 318 \\
\text{Flushing} & 940 & 56 \\
\text{Sunnyside} & 522 & 213 \\
\text{Ditmars} & 763 & 472 \\
\end{array}
\]

- Total development capacity: $C = 4466$

##### Mathematical Model

**Objective Function:**

\[
\max \sum_{i \in \text{Areas}} v_i x_i
\]

**Subject to:**

\[
\sum_{i \in \text{Areas}} w_i x_i \leq 4466
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \text{Areas}
\]

##### Retrieved Information

{
  "capacity": 4466,
  "areas": [
    {"name": "Queens", "Value": 443, "Weight": 104},
    {"name": "Brooklyn", "Value": 522, "Weight": 368},
    {"name": "Manhattan", "Value": 300, "Weight": 483},
    {"name": "Bronx", "Value": 767, "Weight": 165},
    {"name": "Staten Island", "Value": 300, "Weight": 105},
    {"name": "Harlem", "Value": 309, "Weight": 123},
    {"name": "Upper East Side", "Value": 598, "Weight": 131},
    {"name": "Lower Manhattan", "Value": 460, "Weight": 341},
    {"name": "Midtown", "Value": 318, "Weight": 258},
    {"name": "Long Island City", "Value": 126, "Weight": 469},
    {"name": "Williamsburg", "Value": 593, "Weight": 387},
    {"name": "Bushwick", "Value": 871, "Weight": 425},
    {"name": "Flatbush", "Value": 858, "Weight": 482},
    {"name": "Greenpoint", "Value": 321, "Weight": 495},
    {"name": "Park Slope", "Value": 275, "Weight": 305},
    {"name": "Astoria", "Value": 700, "Weight": 377},
    {"name": "Jackson Heights", "Value": 685, "Weight": 318},
    {"name": "Flushing", "Value": 940, "Weight": 56},
    {"name": "Sunnyside", "Value": 522, "Weight": 213},
    {"name": "Ditmars", "Value": 763, "Weight": 472}
  ]
}