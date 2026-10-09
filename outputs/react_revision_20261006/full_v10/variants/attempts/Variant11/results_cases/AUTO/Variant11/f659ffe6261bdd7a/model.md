## Mathematical Model

Sets:
  T = {1, 2, ..., 12}  (months, indexed by t; mapping below)

Parameters (from file_0_view_0):
  Demand_t           = Demand for month t (column "Demand")
  ProductionCost_t   = Unit production cost in month t (column "ProductionCost")
  SetupCost_t        = Setup cost in month t (column "SetupCost")
  HoldingCost_t      = Unit holding cost from month t to t+1 (column "HoldingCost")
  ProductionCapacity_t = Maximum production in month t (column "ProductionCapacity")

Decision variables:
  x_t    ≥ 0   (production quantity in month t, integer or continuous as appropriate)
  inv_t  ≥ 0   (ending inventory after month t)
  y_t ∈ {0,1} (1 if production occurs in month t, 0 otherwise)

Objective:
  Minimize
    ∑_{t∈T} [ ProductionCost_t * x_t + SetupCost_t * y_t + HoldingCost_t * inv_t ]

Subject to:
  Inventory balance:
    inv_0 = 0
    inv_t = inv_{t-1} + x_t - Demand_t      ∀ t ∈ T

  Production capacity and setup linking:
    x_t ≤ ProductionCapacity_t * y_t        ∀ t ∈ T

  No backlogging:
    inv_t ≥ 0                               ∀ t ∈ T

  Zero ending inventory:
    inv_{12} = 0

  Nonnegativity and binary:
    x_t ≥ 0                                 ∀ t ∈ T
    y_t ∈ {0,1}                             ∀ t ∈ T

Data Mapping:
  Set T: All rows in file_0_view_0, with t=1 for "M01", t=2 for "M02", ..., t=12 for "M12"
  Demand_t: column "Demand" in file_0_view_0, row t-1
  ProductionCost_t: column "ProductionCost" in file_0_view_0, row t-1
  SetupCost_t: column "SetupCost" in file_0_view_0, row t-1
  HoldingCost_t: column "HoldingCost" in file_0_view_0, row t-1
  ProductionCapacity_t: column "ProductionCapacity" in file_0_view_0, row t-1

Variables x_t, inv_t, y_t are defined for all t ∈ T = {1,...,12}.