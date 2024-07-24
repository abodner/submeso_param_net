# submeso_param_net
Repo to train a Neural Network parmeterization for submesoscale vertical buoyancy fluxes (Bodner, Balwada, and Zanna, to be submitted to JAMES)

This is the code repository for Bodner, Balwada, and Zanna. A Data-Driven Approach for Parameterizing Ocean Submesoscale  Buoyancy Fluxes (in prep).

* Plotting scripts are in `notebooks`

* Code for preprocessing the LLC4320 to generate training data is under `scripts/preprocess_llc4320/`

* Postprocessing diagnostics are under `scripts/postprocess/`

* Examples for training the CNN under the various test cases discussed in the paper can be found under `scripts/nn_training_examples`. These scripts correspond with the modules found in `submeso_ml` to perscribe the CNN architecture and hyperparameters, as well as the input features, specific regions/seasons, and filter scale.
