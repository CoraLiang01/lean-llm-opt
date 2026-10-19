##### Decision Variables

$x_{ij}^{W}\geq0$: number of white pupils from neighborhood $i\in N$ assigned to school $j\in S$ (continuous).  
$x_{ij}^{NW}\geq0$: number of non‑white pupils from neighborhood $i\in N$ assigned to school $j\in S$ (continuous).  

##### Objective Function

$\displaystyle \min \sum_{i\in N}\sum_{j\in S} d_{ji}\,\bigl(x_{ij}^{W}+x_{ij}^{NW}\bigr)$  

##### Constraints  

1. **Neighborhood supply (white):**  
$\displaystyle \sum_{j\in S}x_{ij}^{W}=W_{i},\quad \forall i\in N$  

2. **Neighborhood supply (non‑white):**  
$\displaystyle \sum_{j\in S}x_{ij}^{NW}=NW_{i},\quad \forall i\in N$  

3. **School capacity:**  
$\displaystyle \sum_{i\in N}\bigl(x_{ij}^{W}+x_{ij}^{NW}\bigr)\leq C_{j},\quad \forall j\in S$  

4. **Racial balance (lower bound):**  
$\displaystyle \sum_{i\in N}x_{ij}^{W}\geq 0.5\,\sum_{i\in N}\bigl(x_{ij}^{W}+x_{ij}^{NW}\bigr),\quad \forall j\in S$  

5. **Racial balance (upper bound):**  
$\displaystyle \sum_{i\in N}x_{ij}^{W}\leq 0.7\,\sum_{i\in N}\bigl(x_{ij}^{W}+x_{ij}^{NW}\bigr),\quad \forall j\in S$  

6. **Non‑negativity:**  
$x_{ij}^{W}\geq0,\;x_{ij}^{NW}\geq0,\quad \forall i\in N,\;j\in S$  

$S=\{I,\,II\}$, $N=\{N01,N02,\dots ,N31\}$, $C_{I}=2028$, $C_{II}=1560$.  
$W_{i}$ and $NW_{i}$ are the white and non‑white populations of neighborhood $i$ (see retrieved data).  
$d_{ji}$ is the distance in miles from school $j$ to neighborhood $i$ (see retrieved data).  
The district white‑student ratio is $0.60$; the racial‑balance constraints enforce a deviation of no more than $10$ percentage points, i.e. $0.50\leq \frac{\sum_{i}x_{ij}^{W}}{\sum_{i}(x_{ij}^{W}+x_{ij}^{NW})}\leq0.70$.

###### Retrieved Information

{
  "schools": ["I","II"],
  "capacities": {"I":2028,"II":1560},
  "neighborhoods": ["N01","N02","N03","N04","N05","N06","N07","N08","N09","N10","N11","N12","N13","N14","N15","N16","N17","N18","N19","N20","N21","N22","N23","N24","N25","N26","N27","N28","N29","N30","N31"],
  "population_white": {
    "N01":78,"N02":57,"N03":47,"N04":78,"N05":57,"N06":47,"N07":78,"N08":57,"N09":46,"N10":77,"N11":56,"N12":46,"N13":77,"N14":56,"N15":46,"N16":77,"N17":56,"N18":46,"N19":77,"N20":56,"N21":46,"N22":77,"N23":56,"N24":46,"N25":77,"N26":56,"N27":46,"N28":77,"N29":56,"N30":46,"N31":74
  },
  "population_nonwhite": {
    "N01":22,"N02":33,"N03":63,"N04":22,"N05":33,"N06":63,"N07":22,"N08":33,"N09":64,"N10":23,"N11":34,"N12":64,"N13":23,"N14":34,"N15":64,"N16":23,"N17":34,"N18":64,"N19":23,"N20":34,"N21":64,"N22":23,"N23":34,"N24":64,"N25":23,"N26":34,"N27":64,"N28":23,"N29":34,"N30":64,"N31":46
  },
  "distance": {
    "I": {
      "N01":1.25,"N02":1.3,"N03":1.35,"N04":1.4,"N05":1.45,"N06":1.5,"N07":1.55,"N08":1.6,"N09":1.65,"N10":1.7,"N11":1.75,"N12":1.8,"N13":1.85,"N14":1.9,"N15":1.95,"N16":2.0,"N17":3.08,"N18":3.16,"N19":3.24,"N20":3.32,"N21":3.4,"N22":3.48,"N23":3.56,"N24":3.64,"N25":3.7199999999999998,"N26":3.8,"N27":3.88,"N28":3.96,"N29":4.04,"N30":4.12,"N31":4.2
    },
    "II": {
      "N01":3.08,"N02":3.16,"N03":3.24,"N04":3.32,"N05":3.4,"N06":3.48,"N07":3.56,"N08":3.64,"N09":3.7199999999999998,"N10":3.8,"N11":3.88,"N12":3.96,"N13":4.04,"N14":4.12,"N15":4.2,"N16":4.28,"N17":1.25,"N18":1.3,"N19":1.35,"N20":1.4,"N21":1.45,"N22":1.5,"N23":1.55,"N24":1.6,"N25":1.65,"N26":1.7,"N27":1.75,"N28":1.8,"N29":1.85,"N30":1.9,"N31":1.95
    }
  },
  "district_white_ratio":0.6,
  "racial_balance_lower":0.5,
  "racial_balance_upper":0.7
}