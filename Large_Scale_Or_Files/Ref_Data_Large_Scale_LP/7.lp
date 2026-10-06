\ Model Sector_Output_Value_Model
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x7342355ebec5ca6f
Minimize
 
Subject To
 Coal_Balance: OutputValue_Coal - 0.4 OutputValue_Power
   - 0.6 OutputValue_Steel = 0
 Power_Balance: - 0.6 OutputValue_Coal + 0.9 OutputValue_Power
   - 0.2 OutputValue_Steel = 0
 Steel_Balance: - 0.4 OutputValue_Coal - 0.5 OutputValue_Power
   + 0.8 OutputValue_Steel = 0
 Assumption: OutputValue_Steel = 10000
Bounds
End
