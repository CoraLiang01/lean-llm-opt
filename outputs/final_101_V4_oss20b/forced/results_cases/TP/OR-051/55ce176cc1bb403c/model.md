##### Decision Variables

$x_{ij}\in \mathbb{Z}_{\ge 0}$: number of units of coffee product $j$ placed in cabinet $i$.

##### Objective Function

$\displaystyle \max \sum_{i\in I}\sum_{j\in J}v_{j}\,x_{ij}$

##### Constraints

1. Cabinet capacity: $\displaystyle \sum_{j\in J}w_{j}\,x_{ij}\leq c_{i},\quad\forall i\in I$  
2. Non‑negativity and integrality: $x_{ij}\in \mathbb{Z}_{\ge 0},\quad\forall i\in I,\;j\in J$  

$I=\{1,2,3,4,5,6,7,8,9,10\}$ (cabinet identifiers)  
$J=\{\text{Espresso Beans},\text{Colombian Roast},\text{Arabica Blend},\text{French Roast},\text{Italian Roast},\text{House Blend},\text{Sumatra Coffee},\text{Mocha Java},\text{Hazelnut Flavor},\text{Caramel Blend},\text{Vanilla Flavor},\text{Cappuccino Mix},\text{Pumpkin Spice},\text{Decaf Roast},\text{Organic Roast},\text{Cold Brew},\text{Peruvian Blend},\text{Kenyan AA}\}$  

$c_{i}$ is the capacity of cabinet $i$ (in weight units).  
$w_{j}$ is the weight of one unit of product $j$.  
$v_{j}$ is the value of one unit of product $j$.

###### Retrieved Information

{
  "cabinets": ["1","2","3","4","5","6","7","8","9","10"],
  "capacities": {"1":400,"2":600,"3":500,"4":700,"5":450,"6":650,"7":550,"8":750,"9":480,"10":520},
  "products": ["Espresso Beans","Colombian Roast","Arabica Blend","French Roast","Italian Roast","House Blend","Sumatra Coffee","Mocha Java","Hazelnut Flavor","Caramel Blend","Vanilla Flavor","Cappuccino Mix","Pumpkin Spice","Decaf Roast","Organic Roast","Cold Brew","Peruvian Blend","Kenyan AA"],
  "values": {"Espresso Beans":100,"Colombian Roast":150,"Arabica Blend":80,"French Roast":120,"Italian Roast":130,"House Blend":110,"Sumatra Coffee":160,"Mocha Java":90,"Hazelnut Flavor":95,"Caramel Blend":105,"Vanilla Flavor":85,"Cappuccino Mix":140,"Pumpkin Spice":75,"Decaf Roast":60,"Organic Roast":170,"Cold Brew":115,"Peruvian Blend":155,"Kenyan AA":125},
  "weights": {"Espresso Beans":1.0,"Colombian Roast":1.5,"Arabica Blend":1.2,"French Roast":1.3,"Italian Roast":1.4,"House Blend":1.1,"Sumatra Coffee":1.8,"Mocha Java":1.2,"Hazelnut Flavor":1.0,"Caramel Blend":1.3,"Vanilla Flavor":1.2,"Cappuccino Mix":1.5,"Pumpkin Spice":1.1,"Decaf Roast":1.0,"Organic Roast":1.6,"Cold Brew":1.4,"Peruvian Blend":1.7,"Kenyan AA":1.3}
}