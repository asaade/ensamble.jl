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
    
      + TAU can either be an empty array or a set of vectors (e.g., `[[0.5, 0.4], ...]`).. If TAU is emplty, an approximation will be estimated from the items in the bank
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

## Example configuration

Here is an example of a configuration for assembling tests using the Test Characteristic Curve (TCC) method. This example uses TCC constraints to control the score distribution across test forms:

### Configuration (TOML) for TCC Optimization

```toml
[FORMS]
NumForms = 5
AnchorTests = 2
ShadowTest = 1

[IRT]
METHOD = "TCC"
THETA = [-3.0, -1.0, 0.0, 1.0, 3.0]
TAU = [[0.5, 0.4, 0.6], [0.6, 0.5, 0.7], [0.7, 0.6, 0.8], [0.8, 0.7, 0.9]]
TAU_INFO = [0.3, 0.4, 0.5, 0.6, 0.7]
RELATIVETARGETWEIGHTS = [1, 1, 1, 1, 1]
RELATIVETARGETPOINTS = [-1.0, 0.0, 1.0]
R = 3
D = 1.0

[FILES]
ITEMSFILE = "data/items.csv"
ANCHORFILE = "data/anchor.csv"
CONSTRAINTSFILE = "data/constraints.csv"
RESULTSFILE = "results/results.csv"
FORMSFILE = "results/forms.csv"
TCCFILE = "results/tcc_output.csv"
PLOTFILE = "results/plot_output.png"
REPORTCATEGORIES = ["CATEGORY_A", "CATEGORY_B"]
REPORTSUMS = ["WORD_COUNT", "IMAGE_COUNT"]

SOLVER = "cplex"
VERBOSE = 1
```

### Example of `items.csv` File Format

The `items.csv` file should contain columns with item parameters, typically including:

  - `ID`: Unique identifier for each item
  - `MODEL_TYPE`: Specifies model type (e.g., "2PL", "3PL", "PCM")
  - `A`: Discrimination parameter
  - `B`: Difficulty (or threshold) parameters, usually as multiple columns `B1`, `B2`, `B3` for polytomous items
  - `C`: Guessing parameter (for "3PL" model)
  - `NUM_CATEGORIES`: Number of response categories for polytomous items

Example `items.csv`:

```csv
ID,MODEL_TYPE,A,B1,B2,B3,C,NUM_CATEGORIES
item1,2PL,1.0,-1.0,,
item2,3PL,0.8,0.5,,0.2
item3,PCM,1.2,0.5,1.0,1.5,,4
item4,GPCM,0.9,0.3,1.2,,
```

This example defines:

  - **Form Configuration**: Configures the number of forms, anchor, and shadow tests.
  - **IRT Settings**: Specifies TCC-based constraints with key parameters such as theta values, target tau means, and tau information.
  - **File Paths**: Specifies paths to required input and output files.

For effective assembly, verify that the items and constraints align with the test blueprint requirements.
