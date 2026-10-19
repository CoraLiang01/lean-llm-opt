##### Decision Variables

$x_{p,a}\geq0$: quantity of product $p\in P$ processed on A‑equipment $a\in A$ (continuous).  
$y_{p,b}\geq0$: quantity of product $p\in P$ processed on B‑equipment $b\in B$ (continuous).

##### Objective Function

$\displaystyle \max \Bigg(\sum_{p\in P}\bigl(\text{unit\_price}_p-\text{raw\_material\_cost}_p\bigr)\sum_{a\in A}x_{p,a}\;-\;\sum_{a\in A}\text{cost}_a\sum_{p\in P}\text{time}_{p,a}\,x_{p,a}\;-\;\sum_{b\in B}\text{cost}_b\sum_{p\in P}\text{time}_{p,b}\,y_{p,b}\Bigg)$

##### Constraints

1. **Flow balance (each unit produced once)**  
   $\displaystyle \sum_{a\in A}x_{p,a}\;=\;\sum_{b\in B}y_{p,b}\quad\forall p\in P$

2. **A‑equipment capacity**  
   $\displaystyle \sum_{p\in P}\text{time}_{p,a}\,x_{p,a}\;\leq\;\text{available\_time}_a\quad\forall a\in A$

3. **B‑equipment capacity**  
   $\displaystyle \sum_{p\in P}\text{time}_{p,b}\,y_{p,b}\;\leq\;\text{available\_time}_b\quad\forall b\in B$

4. **Compatibility (zero for infeasible product‑equipment pairs)**  
   $\displaystyle x_{p,a}=0\quad\text{if }\text{time}_{p,a}\text{ is missing}$  
   $\displaystyle y_{p,b}=0\quad\text{if }\text{time}_{p,b}\text{ is missing}$

$P=\{I,\,II,\,III\}$, $A=\{A1,\,A2,\,A3\}$, $B=\{B1,\,B2,\,B3,\,B4\}$.

##### Retrieved Information

```json
{
  "equipment": ["A1","A2","A3","B1","B2","B3","B4"],
  "available_time": {
    "A1": 6000,
    "A2": 10000,
    "A3": 8000,
    "B1": 4000,
    "B2": 7000,
    "B3": 4000,
    "B4": 5000
  },
  "cost_per_hour": {
    "A1": 300,
    "A2": 321,
    "A3": 203,
    "B1": 250,
    "B2": 783,
    "B3": 200,
    "B4": 300
  },
  "processing_time": {
    "A1": {"I":5,"II":10,"III":null},
    "A2": {"I":7,"II":9,"III":12},
    "A3": {"I":6,"II":11,"III":2},
    "B1": {"I":6,"II":8,"III":null},
    "B2": {"I":4,"II":null,"III":11},
    "B3": {"I":7,"II":null,"III":null},
    "B4": {"I":3,"II":5,"III":8}
  },
  "raw_material_cost": {
    "I": 0.25,
    "II": 0.35,
    "III": 0.5
  },
  "unit_price": {
    "I": 1.25,
    "II": 2,
    "III": 2.8
  }
}
```