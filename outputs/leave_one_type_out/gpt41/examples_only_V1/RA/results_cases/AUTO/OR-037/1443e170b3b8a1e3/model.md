Let the set of vehicle types be indexed by i, corresponding to the ProductName in products.csv. For each vehicle type i, let:

- x_i = number of vehicles of type i to order per day (decision variable, integer, x_i ≥ 0)
- Value_i = profit from selling one unit of vehicle type i (from the Value column)
- Weight_i = inventory space consumed by one unit of vehicle type i (from the Weight column)
- Capacity = overall inventory capacity (from capacity.csv, Capacity = 765)

The mathematical optimization model is:

Variables:
For each i ∈ {1,...,25} (corresponding to the 25 rows in products.csv):
 x_i ∈ {0, 1, 2, ...} (integer, x_i ≥ 0)

Objective:
Maximize total profit:
 maximize 2524 x_Sedan + 4614 x_SUV + 8416 x_Truck + 5917 x_Convertible + 9048 x_Minivan + 1140 x_Coupe + 8962 x_Hatchback + 1888 x_StationWagon + 8487 x_ElectricCar + 4425 x_HybridCar + 4717 x_LuxurySedan + 4210 x_SportsCar + 1226 x_Crossover + 7400 x_DieselTruck + 4639 x_CompactSUV + 7712 x_LuxurySUV + 3299 x_CargoVan + 9895 x_PickupTruck + 4496 x_Roadster + 4526 x_MuscleCar + 5688 x_OffroadVehicle + 3007 x_CamperVan + 3623 x_CompactCar + 8474 x_Motorcycle + 8372 x_ElectricSUV

Subject to:

Inventory capacity constraint:
 99 x_Sedan + 55 x_SUV + 75 x_Truck + 94 x_Convertible + 80 x_Minivan + 82 x_Coupe + 71 x_Hatchback + 100 x_StationWagon + 28 x_ElectricCar + 93 x_HybridCar + 84 x_LuxurySedan + 83 x_SportsCar + 62 x_Crossover + 90 x_DieselTruck + 99 x_CompactSUV + 96 x_LuxurySUV + 21 x_CargoVan + 39 x_PickupTruck + 99 x_Roadster + 81 x_MuscleCar + 6 x_OffroadVehicle + 58 x_CamperVan + 37 x_CompactCar + 15 x_Motorcycle + 37 x_ElectricSUV ≤ 765

x_i ≥ 0 and integer, for all i

Where:
- x_Sedan = number of Sedans to order per day
- x_SUV = number of SUVs to order per day
- x_Truck = number of Trucks to order per day
- x_Convertible = number of Convertibles to order per day
- x_Minivan = number of Minivans to order per day
- x_Coupe = number of Coupes to order per day
- x_Hatchback = number of Hatchbacks to order per day
- x_StationWagon = number of Station Wagons to order per day
- x_ElectricCar = number of Electric Cars to order per day
- x_HybridCar = number of Hybrid Cars to order per day
- x_LuxurySedan = number of Luxury Sedans to order per day
- x_SportsCar = number of Sports Cars to order per day
- x_Crossover = number of Crossovers to order per day
- x_DieselTruck = number of Diesel Trucks to order per day
- x_CompactSUV = number of Compact SUVs to order per day
- x_LuxurySUV = number of Luxury SUVs to order per day
- x_CargoVan = number of Cargo Vans to order per day
- x_PickupTruck = number of Pickup Trucks to order per day
- x_Roadster = number of Roadsters to order per day
- x_MuscleCar = number of Muscle Cars to order per day
- x_OffroadVehicle = number of Off-road Vehicles to order per day
- x_CamperVan = number of Camper Vans to order per day
- x_CompactCar = number of Compact Cars to order per day
- x_Motorcycle = number of Motorcycles to order per day
- x_ElectricSUV = number of Electric SUVs to order per day

This is a 0-1 knapsack-type integer program, maximizing profit subject to the total inventory space not exceeding 765 units.