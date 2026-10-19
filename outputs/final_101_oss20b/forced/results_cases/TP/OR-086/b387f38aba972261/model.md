##### Decision Variables

$x_{g,b}\geq0$: quantity of raw grade $g\in\{I,II,III\}$ allocated to wine brand $b\in\{Red,Yellow,Blue\}$ (continuous).

##### Objective Function

$\max\sum_{b\in\{Red,Yellow,Blue\}}\sum_{g\in\{I,II,III\}}SellingPrice_{b}\,x_{g,b}\;-\;\sum_{g\in\{I,II,III\}}\sum_{b\in\{Red,Yellow,Blue\}}Cost_{g}\,x_{g,b}$

##### Constraints

1. **Blending Requirements – Red**  
   $x_{I,Red}\leq0.10\,(x_{I,Red}+x_{II,Red}+x_{III,Red})$  
   $x_{II,Red}\geq0.50\,(x_{I,Red}+x_{II,Red}+x_{III,Red})$

2. **Blending Requirements – Yellow**  
   $x_{III,Yellow}\leq0.70\,(x_{I,Yellow}+x_{II,Yellow}+x_{III,Yellow})$  
   $x_{I,Yellow}\geq0.20\,(x_{I,Yellow}+x_{II,Yellow}+x_{III,Yellow})$

3. **Blending Requirements – Blue**  
   $x_{I,Blue}\leq0.50\,(x_{I,Blue}+x_{II,Blue}+x_{III,Blue})$  
   $x_{II,Blue}\geq0.10\,(x_{I,Blue}+x_{II,Blue}+x_{III,Blue})$

4. **Raw Material Supply Constraints**  
   $x_{I,Red}+x_{I,Yellow}+x_{I,Blue}\leq1500$  
   $x_{II,Red}+x_{II,Yellow}+x_{II,Blue}\leq2000$  
   $x_{III,Red}+x_{III,Yellow}+x_{III,Blue}\leq1000$

5. **Minimum Production for Red**  
   $x_{I,Red}+x_{II,Red}+x_{III,Red}\geq2000$

6. **Non‑negativity**  
   $x_{g,b}\geq0\quad\forall g\in\{I,II,III\},\;b\in\{Red,Yellow,Blue\}$

##### Retrieved Information

{
  "grades": ["I","II","III"],
  "brands": ["Red","Yellow","Blue"],
  "supply_limit": {
    "I": 1500,
    "II": 2000,
    "III": 1000
  },
  "cost": {
    "I": 6,
    "II": 4.5,
    "III": 3
  },
  "selling_price": {
    "Red": 5.5,
    "Yellow": 5,
    "Blue": 4.8
  },
  "blending_requirements": {
    "Red": {
      "I_upper": 0.10,
      "II_lower": 0.50
    },
    "Yellow": {
      "III_upper": 0.70,
      "I_lower": 0.20
    },
    "Blue": {
      "I_upper": 0.50,
      "II_lower": 0.10
    }
  }
}