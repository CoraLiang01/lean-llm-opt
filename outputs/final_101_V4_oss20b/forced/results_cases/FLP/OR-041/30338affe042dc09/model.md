##### Decision Variables

$x_i \geq 0$: scale of development per day in area $i \in I$ (continuous).

##### Objective Function

$\max \sum_{i\in I} v_i x_i$

##### Constraints

1. Capacity constraint: $\sum_{i\in I} w_i x_i \leq C$.
2. Non‑negativity: $x_i \geq 0,\quad \forall i\in I$.

##### Parameters

$I=\{Queens,Brooklyn,Manhattan,Bronx,Staten\ Island,Harlem,Upper\ East\ Side,Lower\ Manhattan,Midtown,Long\ Island\ City,Williamsburg,Bushwick,Flatbush,Greenpoint,Park\ Slope,Astoria,Jackson\ Heights,Flushing,Sunnyside,Ditmars\}$  

$C=586$  

$v=\begin{bmatrix}
469\\
290\\
236\\
235\\
745\\
684\\
444\\
172\\
1000\\
336\\
546\\
535\\
539\\
831\\
139\\
432\\
627\\
629\\
292\\
978
\end{bmatrix}$  

$w=\begin{bmatrix}
954\\
650\\
961\\
950\\
379\\
776\\
381\\
808\\
937\\
608\\
912\\
391\\
465\\
490\\
918\\
787\\
347\\
274\\
642\\
130
\end{bmatrix}$  

###### Retrieved Information

{
  "capacity": 586,
  "areas": [
    {"ProductName":"Queens","Value":469,"Weight":954},
    {"ProductName":"Brooklyn","Value":290,"Weight":650},
    {"ProductName":"Manhattan","Value":236,"Weight":961},
    {"ProductName":"Bronx","Value":235,"Weight":950},
    {"ProductName":"Staten Island","Value":745,"Weight":379},
    {"ProductName":"Harlem","Value":684,"Weight":776},
    {"ProductName":"Upper East Side","Value":444,"Weight":381},
    {"ProductName":"Lower Manhattan","Value":172,"Weight":808},
    {"ProductName":"Midtown","Value":1000,"Weight":937},
    {"ProductName":"Long Island City","Value":336,"Weight":608},
    {"ProductName":"Williamsburg","Value":546,"Weight":912},
    {"ProductName":"Bushwick","Value":535,"Weight":391},
    {"ProductName":"Flatbush","Value":539,"Weight":465},
    {"ProductName":"Greenpoint","Value":831,"Weight":490},
    {"ProductName":"Park Slope","Value":139,"Weight":918},
    {"ProductName":"Astoria","Value":432,"Weight":787},
    {"ProductName":"Jackson Heights","Value":627,"Weight":347},
    {"ProductName":"Flushing","Value":629,"Weight":274},
    {"ProductName":"Sunnyside","Value":292,"Weight":642},
    {"ProductName":"Ditmars","Value":978,"Weight":130}
  ]
}