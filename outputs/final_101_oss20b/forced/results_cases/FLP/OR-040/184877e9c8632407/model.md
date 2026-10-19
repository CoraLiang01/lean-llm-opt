##### Decision Variables

$x_i \in \mathbb{Z}_{\ge 0}$: development scale in area $i \in I$.

##### Objective Function

$\displaystyle \max \sum_{i\in I} v_i\,x_i$

##### Constraints

1. Capacity: $\displaystyle \sum_{i\in I} w_i\,x_i \;\le\; 4466$

2. Domains: $\displaystyle x_i \in \mathbb{Z}_{\ge 0},\;\forall i\in I$

##### Parameters

$I=\{\text{Queens},\text{Brooklyn},\text{Manhattan},\text{Bronx},\text{Staten Island},\text{Harlem},\text{Upper East Side},\text{Lower Manhattan},\text{Midtown},\text{Long Island City},\text{Williamsburg},\text{Bushwick},\text{Flatbush},\text{Greenpoint},\text{Park Slope},\text{Astoria},\text{Jackson Heights},\text{Flushing},\text{Sunnyside},\text{Ditmars}\}$  

$v=\begin{bmatrix}
443\\
522\\
300\\
767\\
300\\
309\\
598\\
460\\
318\\
126\\
593\\
871\\
858\\
321\\
275\\
700\\
685\\
940\\
522\\
763
\end{bmatrix}$  

$w=\begin{bmatrix}
104\\
368\\
483\\
165\\
105\\
123\\
131\\
341\\
258\\
469\\
387\\
425\\
482\\
495\\
305\\
377\\
318\\
56\\
213\\
472
\end{bmatrix}$  

Capacity: $C=4466$