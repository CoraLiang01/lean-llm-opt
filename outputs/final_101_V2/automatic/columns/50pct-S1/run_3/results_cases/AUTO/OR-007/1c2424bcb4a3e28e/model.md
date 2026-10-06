Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the following vehicle types as given by the ProductName column.

##### Sets and Parameters

Let $I$ be the set of vehicle types (in source order):

\[
\begin{array}{ll}
\text{Sedan} & (\text{Value}=2524,\, \text{Weight}=99) \\
\text{SUV} & (4614,\, 55) \\
\text{Truck} & (8416,\, 75) \\
\text{Convertible} & (5917,\, 94) \\
\text{Minivan} & (9048,\, 80) \\
\text{Coupe} & (1140,\, 82) \\
\text{Hatchback} & (8962,\, 71) \\
\text{Station Wagon} & (1888,\, 100) \\
\text{Electric Car} & (8487,\, 28) \\
\text{Hybrid Car} & (4425,\, 93) \\
\text{Luxury Sedan} & (4717,\, 84) \\
\text{Sports Car} & (4210,\, 83) \\
\text{Crossover} & (1226,\, 62) \\
\text{Diesel Truck} & (7400,\, 90) \\
\text{Compact SUV} & (4639,\, 99) \\
\text{Luxury SUV} & (7712,\, 96) \\
\text{Cargo Van} & (3299,\, 21) \\
\text{Pickup Truck} & (9895,\, 39) \\
\text{Roadster} & (4496,\, 99) \\
\text{Muscle Car} & (4526,\, 81) \\
\text{Off-road Vehicle} & (5688,\, 6) \\
\text{Camper Van} & (3007,\, 58) \\
\text{Compact Car} & (3623,\, 37) \\
\text{Motorcycle} & (8474,\, 15) \\
\text{Electric SUV} & (8372,\, 37) \\
\end{array}
\]

Let $v_i$ be the Value (profit) for vehicle type $i$.

Let $w_i$ be the Weight (stock space required) for vehicle type $i$.

Let $C$ be the overall inventory capacity limit (not provided in the data above; must be specified from 'capacity.csv'—if not, leave as $C$).

##### Decision Variables

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

##### Objective Function

\[
\max \sum_{i \in I} v_i x_i
\]

##### Constraints

\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \geq 0 \text{ and integer} \quad \forall i \in I
\]

##### Complete Model

\[
\begin{align*}
\max \quad & 2524\,x_{\text{Sedan}} + 4614\,x_{\text{SUV}} + 8416\,x_{\text{Truck}} + 5917\,x_{\text{Convertible}} + 9048\,x_{\text{Minivan}} + 1140\,x_{\text{Coupe}} + 8962\,x_{\text{Hatchback}} \\
& + 1888\,x_{\text{Station Wagon}} + 8487\,x_{\text{Electric Car}} + 4425\,x_{\text{Hybrid Car}} + 4717\,x_{\text{Luxury Sedan}} + 4210\,x_{\text{Sports Car}} + 1226\,x_{\text{Crossover}} \\
& + 7400\,x_{\text{Diesel Truck}} + 4639\,x_{\text{Compact SUV}} + 7712\,x_{\text{Luxury SUV}} + 3299\,x_{\text{Cargo Van}} + 9895\,x_{\text{Pickup Truck}} + 4496\,x_{\text{Roadster}} \\
& + 4526\,x_{\text{Muscle Car}} + 5688\,x_{\text{Off-road Vehicle}} + 3007\,x_{\text{Camper Van}} + 3623\,x_{\text{Compact Car}} + 8474\,x_{\text{Motorcycle}} + 8372\,x_{\text{Electric SUV}} \\
\text{s.t.} \quad & 99\,x_{\text{Sedan}} + 55\,x_{\text{SUV}} + 75\,x_{\text{Truck}} + 94\,x_{\text{Convertible}} + 80\,x_{\text{Minivan}} + 82\,x_{\text{Coupe}} + 71\,x_{\text{Hatchback}} \\
& + 100\,x_{\text{Station Wagon}} + 28\,x_{\text{Electric Car}} + 93\,x_{\text{Hybrid Car}} + 84\,x_{\text{Luxury Sedan}} + 83\,x_{\text{Sports Car}} + 62\,x_{\text{Crossover}} \\
& + 90\,x_{\text{Diesel Truck}} + 99\,x_{\text{Compact SUV}} + 96\,x_{\text{Luxury SUV}} + 21\,x_{\text{Cargo Van}} + 39\,x_{\text{Pickup Truck}} + 99\,x_{\text{Roadster}} \\
& + 81\,x_{\text{Muscle Car}} + 6\,x_{\text{Off-road Vehicle}} + 58\,x_{\text{Camper Van}} + 37\,x_{\text{Compact Car}} + 15\,x_{\text{Motorcycle}} + 37\,x_{\text{Electric SUV}} \leq C \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\end{align*}
\]

where $C$ is the overall inventory capacity limit (to be specified from 'capacity.csv').