### Configuration File Documentation

The TOML configuration file defines parameters for test assembly settings, item constraints, file paths, and solver options. It consists of several key sections:

#### `[FORMS]`

- **NumForms**: Sets the number of forms to be assembled.
- **AnchorTests**: Specifies the number of anchor forms, used cycling across all forms to ensure consistency.
- **ShadowTest**: Indicates the use of a 'shadow test' for iterative assembly. Shadow tests are used as an heuristic method to maintain constraints across multiple assembly cycles by reserving items for future forms.

#### `[IRT]`

- **METHOD**: Specifies the method for scoring or matching (e.g., `TCC2`). Determines the model's scoring approach.
- **THETA**: Ability level values to match expected scores and other metrics. I set of 3..5 well targeted theta points in the curve are often enough to match the compete range.
- **TAU** and **TAU_INFO**: Arrays for target means and variances at specified theta levels for characteristic and information curves.
  - TAU can either be an empty array or a set of vectors (e.g., `[[0.5, 0.4], ...]`).. If TAU is emplty, an approximation will be estimated from the items in the bank
- **RELATIVETARGETWEIGHTS** and **RELATIVETARGETPOINTS**: Lists of weights and target points for information or score matching, providing flexibility in test design goals.
- **R**: The maximum power for item probabilities of correct response for use using the TCC method, folowing van del Linden's observed scores "local equating" method.
- **D**: Scaling constant for IRT logistic models (typically `1.0` for standard logistic models and 1.7 to approximate a normat curve).

#### `[FILES]`

- **ITEMSFILE**: Path to the CSV file with item parameters and attributes.
- **ANCHORFILE**: Path to the anchor items CSV, if applicable.
- **CONSTRAINTSFILE**: Path to the constraints CSV, specifying the conditions items must satisfy.
- **RESULTSFILE**: Path for outputting final test assembly results.
- **FORMSFILE**: Path to save individual form compositions.
- **TCCFILE**: Output file for Test Characteristic Curve (TCC) data.
- **PLOTFILE**: Path to save visual plots for test information and TCC data.
- **REPORTCATEGORIES** and **REPORTSUMS**: Lists of item attributes to include in reporting (e.g., categories for content analysis and sums for item attributes).

#### Solver and Debugging

- **SOLVER**: Name of the solver (e.g., `cplex`) used for optimization.
- **VERBOSE**: Level of verbosity for logging and debugging, where `0` is silent, and higher values increase detail in the output.
