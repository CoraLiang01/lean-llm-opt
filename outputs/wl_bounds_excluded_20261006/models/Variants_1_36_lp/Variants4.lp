\ Canonical native-bound formulation; bounds excluded from WL graph.
Minimize
  obj: + 12 y_SC1 + 15 y_SC2 + 18 y_SC3 + 10 y_SC4 + 14 y_SC5 + 13 y_SC6 + 16 y_SC7 + 11 y_SC8
Subject To
  cover_D1: + 1 y_SC1 + 1 y_SC7 >= 1
  cover_D2: + 1 y_SC1 + 1 y_SC2 >= 1
  cover_D3: + 1 y_SC2 + 1 y_SC8 >= 1
  cover_D4: + 1 y_SC1 + 1 y_SC3 + 1 y_SC8 >= 1
  cover_D5: + 1 y_SC2 + 1 y_SC3 >= 1
  cover_D6: + 1 y_SC3 + 1 y_SC4 >= 1
  cover_D7: + 1 y_SC4 + 1 y_SC5 >= 1
  cover_D8: + 1 y_SC5 + 1 y_SC6 + 1 y_SC8 >= 1
  cover_D9: + 1 y_SC6 + 1 y_SC7 >= 1
  cover_D10: + 1 y_SC5 + 1 y_SC7 >= 1
Bounds
  0 <= y_SC1 <= 1
  0 <= y_SC2 <= 1
  0 <= y_SC3 <= 1
  0 <= y_SC4 <= 1
  0 <= y_SC5 <= 1
  0 <= y_SC6 <= 1
  0 <= y_SC7 <= 1
  0 <= y_SC8 <= 1
Binaries
  y_SC1 y_SC2 y_SC3 y_SC4 y_SC5 y_SC6 y_SC7 y_SC8
End
