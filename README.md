# When the Universe is Too Big: Bounding Consideration Probabilities for Plackett--Luce Rankings

## Code Supplement

### Files in this repository:

- `logit.py` create and fit a Plackett-Luce logit model. Written by Kiran Tomlinson.
- `data_utils.py` utility functions to read and process the state narcissism data.
- `calculate_bounds.py` generate baseline and tightened consideration probability bounds.
- `state_bounds_demo.py` replicates the calculations of consideration probability bounds for the 50 U.S. states from the Putnam dataset.

### Instructions to replicate bounds on U.S. state consideration probabilities from section 7: 

1. Copy `State Narcissism Scrubbed Data.csv` from [Putnam et al, 2018](https://osf.io/tnjqs/)  into `data/`
2. Run `state_bounds_demo.py` to display initial and tightened lower and upper bounds on consideration probabilities for each state, sorted in descending order of final upper bound.

### Data from: 

Adam L Putnam, Morgan Q Ross, Laura K Soter, and Henry L Roediger III. Collective narcissism: Americans exaggerate the role of their home state in appraising us history. Psychological Science, 29(9):1414–1422, 2018
