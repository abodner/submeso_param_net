# submeso_param_net
Repo to train a Neural Network parmeterization for submesoscale vertical buoyancy fluxes (Bodner, Balwada, and Zanna, to be submitted to JAMES)

This is the code repository for Bodner, Balwada, and Zanna. A Data-Driven Approach for Parameterizing Ocean Submesoscale  Buoyancy Fluxes (in prep).

Plotting scripts are in the folder notebooks/plotting

Code for processing the LLC4320 to generate training data is under scripts/preprocess_llc4320

The data loader is located in submeso_ml/dadataset.py 
The CNN recieved parameters from ** and calls submeso_ml/models/fcnn.py and submeso_ml/systems/regression_system.py during training.
