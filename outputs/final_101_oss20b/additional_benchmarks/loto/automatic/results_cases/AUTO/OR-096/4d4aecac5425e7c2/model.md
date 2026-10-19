##### Decision Variables

$x_{s,n,w}\geq0$: quantity of white students assigned from school $s\in S$ to neighborhood $n\in N$ (continuous).  
$x_{s,n,nw}\geq0$: quantity of non‑white students assigned from school $s\in S$ to neighborhood $n\in N$ (continuous).

##### Objective Function

$\displaystyle \min \sum_{s\in S}\sum_{n\in N}d_{s,n}\,\bigl(x_{s,n,w}+x_{s,n,nw}\bigr)$

##### Constraints

1. **Neighborhood population (white):**  
   $\displaystyle \sum_{s\in S}x_{s,n,w}=P_{n,w},\quad\forall n\in N$  

2. **Neighborhood population (non‑white):**  
   $\displaystyle \sum_{s\in S}x_{s,n,nw}=P_{n,nw},\quad\forall n\in N$  

3. **School capacity:**  
   $\displaystyle \sum_{n\in N}\bigl(x_{s,n,w}+x_{s,n,nw}\bigr)\leq C_{s},\quad\forall s\in S$  

4. **Racial balance (lower bound):**  
   $\displaystyle \sum_{n\in N}x_{s,n,w}\;\geq\;\sum_{n\in N}x_{s,n,nw},\quad\forall s\in S$  

5. **Racial balance (upper bound):**  
   $\displaystyle \sum_{n\in N}x_{s,n,w}\;\leq\;\frac{7}{3}\,\sum_{n\in N}x_{s,n,nw},\quad\forall s\in S$  

6. **Non‑negativity:**  
   $x_{s,n,w}\geq0,\;x_{s,n,nw}\geq0,\quad\forall s\in S,\;n\in N$  

$S=\{I,\,II\}$, $N=\{N01,N02,\dots,N31\}$.

##### Retrieved Information

```json
{
  "schools": [
    {"School":"I"},
    {"School":"II"}
  ],
  "capacities": [
    {"School":"I","Capacity":2028},
    {"School":"II","Capacity":1560}
  ],
  "neighborhoods": [
    {"Neighborhood":"N01","Population_White":78,"Population_NonWhite":22},
    {"Neighborhood":"N02","Population_White":57,"Population_NonWhite":33},
    {"Neighborhood":"N03","Population_White":47,"Population_NonWhite":63},
    {"Neighborhood":"N04","Population_White":78,"Population_NonWhite":22},
    {"Neighborhood":"N05","Population_White":57,"Population_NonWhite":33},
    {"Neighborhood":"N06","Population_White":47,"Population_NonWhite":63},
    {"Neighborhood":"N07","Population_White":78,"Population_NonWhite":22},
    {"Neighborhood":"N08","Population_White":57,"Population_NonWhite":33},
    {"Neighborhood":"N09","Population_White":46,"Population_NonWhite":64},
    {"Neighborhood":"N10","Population_White":77,"Population_NonWhite":23},
    {"Neighborhood":"N11","Population_White":56,"Population_NonWhite":34},
    {"Neighborhood":"N12","Population_White":46,"Population_NonWhite":64},
    {"Neighborhood":"N13","Population_White":77,"Population_NonWhite":23},
    {"Neighborhood":"N14","Population_White":56,"Population_NonWhite":34},
    {"Neighborhood":"N15","Population_White":46,"Population_NonWhite":64},
    {"Neighborhood":"N16","Population_White":77,"Population_NonWhite":23},
    {"Neighborhood":"N17","Population_White":56,"Population_NonWhite":34},
    {"Neighborhood":"N18","Population_White":46,"Population_NonWhite":64},
    {"Neighborhood":"N19","Population_White":77,"Population_NonWhite":23},
    {"Neighborhood":"N20","Population_White":56,"Population_NonWhite":34},
    {"Neighborhood":"N21","Population_White":46,"Population_NonWhite":64},
    {"Neighborhood":"N22","Population_White":77,"Population_NonWhite":23},
    {"Neighborhood":"N23","Population_White":56,"Population_NonWhite":34},
    {"Neighborhood":"N24","Population_White":46,"Population_NonWhite":64},
    {"Neighborhood":"N25","Population_White":77,"Population_NonWhite":23},
    {"Neighborhood":"N26","Population_White":56,"Population_NonWhite":34},
    {"Neighborhood":"N27","Population_White":46,"Population_NonWhite":64},
    {"Neighborhood":"N28","Population_White":77,"Population_NonWhite":23},
    {"Neighborhood":"N29","Population_White":56,"Population_NonWhite":34},
    {"Neighborhood":"N30","Population_White":46,"Population_NonWhite":64},
    {"Neighborhood":"N31","Population_White":74,"Population_NonWhite":46}
  ],
  "distances": [
    {
      "School":"I",
      "N01":1.25,"N02":1.3,"N03":1.35,"N04":1.4,"N05":1.45,"N06":1.5,"N07":1.55,"N08":1.6,"N09":1.65,"N10":1.7,"N11":1.75,"N12":1.8,"N13":1.85,"N14":1.9,"N15":1.95,"N16":2.0,"N17":3.08,"N18":3.16,"N19":3.24,"N20":3.32,"N21":3.4,"N22":3.48,"N23":3.56,"N24":3.64,"N25":3.7199999999999998,"N26":3.8,"N27":3.88,"N28":3.96,"N29":4.04,"N30":4.12,"N31":4.2
    },
    {
      "School":"II",
      "N01":3.08,"N02":3.16,"N03":3.24,"N04":3.32,"N05":3.4,"N06":3.48,"N07":3.56,"N08":3.64,"N09":3.7199999999999998,"N10":3.8,"N11":3.88,"N12":3.96,"N13":4.04,"N14":4.12,"N15":4.2,"N16":4.28,"N17":1.25,"N18":1.3,"N19":1.35,"N20":1.4,"N21":1.45,"N22":1.5,"N23":1.55,"N24":1.6,"N25":1.65,"N26":1.7,"N27":1.75,"N28":1.8,"N29":1.85,"N30":1.9,"N31":1.95
    }
  ]
}
```