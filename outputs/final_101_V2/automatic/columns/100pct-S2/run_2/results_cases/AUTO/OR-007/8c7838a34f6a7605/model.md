##### Sets and Indices

Let $I$ be the set of vehicle types (indexed by $i$), as given by the ProductName column in products.csv.

##### Parameters

- $p_i$: Profit (Value) for vehicle type $i$ (from products.csv)
- $a_i$: 1 for all $i$ (each vehicle counts as one unit toward capacity)
- $C$: Overall inventory capacity (from capacity.csv, Capacity column)

##### Decision Variables

- $x_i$: Number of vehicles of type $i$ to order per day ($x_i \in \mathbb{Z}_{\geq 0}$)

##### Mathematical Model

Objective:
\[
\max \sum_{i \in I} p_i x_i
\]

Subject to:
\[
\sum_{i \in I} x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

##### Numerical Formulation

Let $I$ = 
{ Sedan, SUV, Truck, Convertible, Minivan, Coupe, Hatchback, Station Wagon, Electric Car, Hybrid Car, Luxury Sedan, Sports Car, Crossover, Diesel Truck, Compact SUV, Luxury SUV, Cargo Van, Pickup Truck, Roadster, Muscle Car, Off-road Vehicle, Camper Van, Compact Car, Motorcycle, Electric SUV }

Profits ($p_i$):

- Sedan: 2524
- SUV: 4614
- Truck: 8416
- Convertible: 5917
- Minivan: 9048
- Coupe: 1140
- Hatchback: 8962
- Station Wagon: 1888
- Electric Car: 8487
- Hybrid Car: 4425
- Luxury Sedan: 4717
- Sports Car: 4210
- Crossover: 1226
- Diesel Truck: 7400
- Compact SUV: 4639
- Luxury SUV: 7712
- Cargo Van: 3299
- Pickup Truck: 9895
- Roadster: 4496
- Muscle Car: 4526
- Off-road Vehicle: 5688
- Camper Van: 3007
- Compact Car: 3623
- Motorcycle: 8474
- Electric SUV: 8372

Capacity ($C$): 765

Decision variables: $x_i \in \mathbb{Z}_{\geq 0}$ for each $i$ above.

Objective:
\[
\max \big(
2524\,x_{\text{Sedan}} + 4614\,x_{\text{SUV}} + 8416\,x_{\text{Truck}} + 5917\,x_{\text{Convertible}} + 9048\,x_{\text{Minivan}} + 1140\,x_{\text{Coupe}} + 8962\,x_{\text{Hatchback}} + 1888\,x_{\text{Station Wagon}} + 8487\,x_{\text{Electric Car}} + 4425\,x_{\text{Hybrid Car}} + 4717\,x_{\text{Luxury Sedan}} + 4210\,x_{\text{Sports Car}} + 1226\,x_{\text{Crossover}} + 7400\,x_{\text{Diesel Truck}} + 4639\,x_{\text{Compact SUV}} + 7712\,x_{\text{Luxury SUV}} + 3299\,x_{\text{Cargo Van}} + 9895\,x_{\text{Pickup Truck}} + 4496\,x_{\text{Roadster}} + 4526\,x_{\text{Muscle Car}} + 5688\,x_{\text{Off-road Vehicle}} + 3007\,x_{\text{Camper Van}} + 3623\,x_{\text{Compact Car}} + 8474\,x_{\text{Motorcycle}} + 8372\,x_{\text{Electric SUV}}
\big)
\]

Subject to:
\[
x_{\text{Sedan}} + x_{\text{SUV}} + x_{\text{Truck}} + x_{\text{Convertible}} + x_{\text{Minivan}} + x_{\text{Coupe}} + x_{\text{Hatchback}} + x_{\text{Station Wagon}} + x_{\text{Electric Car}} + x_{\text{Hybrid Car}} + x_{\text{Luxury Sedan}} + x_{\text{Sports Car}} + x_{\text{Crossover}} + x_{\text{Diesel Truck}} + x_{\text{Compact SUV}} + x_{\text{Luxury SUV}} + x_{\text{Cargo Van}} + x_{\text{Pickup Truck}} + x_{\text{Roadster}} + x_{\text{Muscle Car}} + x_{\text{Off-road Vehicle}} + x_{\text{Camper Van}} + x_{\text{Compact Car}} + x_{\text{Motorcycle}} + x_{\text{Electric SUV}} \leq 765
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]