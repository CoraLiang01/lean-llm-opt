##### Decision Variables

$x_i \geq 0$: number of units of FAUX product $i$ that will be fulfilled, for $i \in I$.

##### Objective Function

$\displaystyle \max \sum_{i\in I} r_i x_i$

##### Constraints

1. Demand satisfaction: $\displaystyle x_i \leq d_i,\quad \forall i\in I$  
2. Inventory limitation: $\displaystyle x_i \leq I_i,\quad \forall i\in I$  
3. Non‑negativity: $\displaystyle x_i \geq 0,\quad \forall i\in I$

##### Parameters

$I=\{P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12\}$  

$P1:$ FAUX FUR JEWEL SWEATER  
$P2:$ FAUX LEATHER BOMBER JACKET  
$P3:$ FAUX LEATHER BOXY FIT JACKET  
$P4:$ FAUX LEATHER JACKET  
$P5:$ FAUX LEATHER OVERSIZED JACKET LIMITED EDITION  
$P6:$ FAUX LEATHER PUFFER JACKET  
$P7:$ FAUX SHEARLING LINED SUEDE BOOTS  
$P8:$ FAUX SHEARLING PLAID JACKET  
$P9:$ FAUX SUEDE BOMBER JACKET  
$P10:$ FAUX SUEDE JACKET  
$P11:$ FAUX SUEDE OVERSHIRT  
$P12:$ FAUX SUEDE PATCH JACKET  

Demand vector $d = [3025,\; 9585,\; 4486,\; 10322,\; 4868,\; 8482,\; 2607,\; 1784,\; 6626,\; 3256,\; 2955,\; 910]$  

Initial inventory vector $I = [20970,\; 71970,\; 32730,\; 71130,\; 34910,\; 64010,\; 20760,\; 12490,\; 50300,\; 24570,\; 24430,\; 7070]$  

Revenue vector $r = [35.9,\; 69.9,\; 99.9,\; 99.9,\; 159.0,\; 69.99,\; 99.9,\; 89.9,\; 69.9,\; 89.9,\; 69.9,\; 89.9]$