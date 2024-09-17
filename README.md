# ensamble.jl

Automatic Test Assembly (ATA)  in julia and a MIP solver

## The Software

This repository presents an example of how the process of selecting items for a test may be implemented using free software. For now, it is only an early and incomplete example and is not recommended for use in high-stakes applications.

# Ensamble.jl

**Automatic Test Assembly (ATA) in Julia using Mixed Integer Programming (MIP)**

## Overview

For example, if a psychometrician needs to create multiple test forms that adhere to certain constraints, such as maintaining a balance of content areas and difficulty levels, Ensamble.jl automates this process and ensures all constraints are met.

In psychometrics, standardized tests often rely on a single version of a test, created after extensive research. However, reusing the same test repeatedly is impractical in educational contexts, and assembling multiple test forms is exponentially more complex. Ensamble.jl automates this process by ensuring content balance and comparable difficulty levels across forms, addressing issues of fairness and equity for test-takers.

Traditionally, test assembly has been a manual, iterative task, requiring the careful selection of items to meet specifications. With the rise of optimization techniques, this process has been automated, reducing time and errors. Ensamble.jl uses Item Response Theory (IRT) to predict item performance, ensuring consistency across test forms.

Julia is ideal for this purpose due to its speed and simplicity, while JuMP provides robust tools for formulating optimization models, working seamlessly with various solvers.

Ensamble.jl is a Julia-based system designed for **Automatic Test Assembly (ATA)**, focusing on efficiency, objectivity, and flexibility. It uses dichotomic  **Item Response Theory (IRT)** models to generate multiple test forms that are comparable in content and difficulty while satisfying specific constraints, such as content balance and item usage.

The system leverages **Mixed Integer Programming (MIP)** solvers to find the optimal combination of test items, making it a powerful tool for large-scale testing programs, certification exams, and adaptive testing systems. With support for various solvers, it is versatile and adaptable to different use cases.

**Key Goals**:
- **Efficiency**: Automate the test assembly process.
- **Fairness**: Ensure consistency and fairness in test content and difficulty.
- **Customization**: Provide flexible constraint definitions to meet specific testing needs.

---

## Why Automatic Test Assembly?

Manual test assembly is both **time-consuming** and **prone to errors**, especially when generating multiple test forms with comparable content. Ensamble.jl addresses these challenges by:
- **Automating the selection of test items** based on predefined constraints.
- **Ensuring fairness and balance** by using strict test specifications to meet the content needs and Item Response Theory (IRT) to standardize scores across forms.
- **Providing flexibility** in constraint handling, allowing for tests with varying content and statistical properties.


## Alternatives

There are other ways to achieve this. In R, for example, the package TestDesign seems to be a good solution that saves several steps and requires little programming. Other examples include eatATA, ATA, xxIRT, dexterMST, catR, mstR —all of them in R, perhaps the most popular language for this purpose. Some of these packages are designed for assembling adaptive tests.

In Julia, Python, and SAS, there are interesting, though somewhat unpolished, solutions that require at least basic knowledge of the underlying programming languages. In a way, these can be considered experimental libraries. Major testing and assessment agencies typically develop their own in-house solutions.


## Structure

.
├── data
│  ├── anchor.csv
│  ├── config.toml
│  ├── constraints.csv
│  ├── items.csv
│  ├── model.lp
│  └── solver_config.toml
├── debug.log
├── docs
│  └── structure.md
├── items.csv
├── LICENSE
├── logfile.log
├── Manifest.toml
├── Project.toml
├── README.md
├── results
│  ├── combined_plot.pdf
│  ├── model.lp
│  ├── p1_characteristic_curves.svg
│  ├── p2_information_curves.svg
│  ├── p3_1_observed_scores_n01.svg
│  ├── p3_2_observed_scores_n_variations.svg
│  ├── results.csv
│  └── tcc.csv
├── src
│  ├── config
│  │  ├── assembly_config_loader.jl
│  │  ├── bank_data_loader.jl
│  │  ├── config_loader.jl
│  │  └── irt_data_loader.jl
│  ├── configuration.jl
│  ├── constants.jl
│  ├── debug.log
│  ├── display
│  │  ├── charts.jl
│  │  └── display_results.jl
│  ├── ensamble.jl
│  ├── logfile.log
│  ├── model
│  │  ├── constraints.jl
│  │  ├── criteria_parser.jl
│  │  ├── model_initializer.jl
│  │  └── solvers.jl
│  └── utils
│     ├── custom_logger.jl
│     ├── stats_functions.jl
│     └── string_utils.jl
└── tests


---

## How It Works

Ensamble.jl is built around the following components:

1. **Item Response Theory (IRT)**: IRT provides a framework for modeling item parameters such as difficulty (b), discrimination (a), and guessing (c). These parameters are used to predict how well each item will perform across different ability levels.

2. **Mixed Integer Programming (MIP)**: MIP optimization is used to select the best set of items that meet the defined constraints. These constraints can include:
   - Content area balance
   - Maximum or minimum test length
   - Item overlap between forms
   - Anchor item usage
   - Statistical properties such as item information and expected score

3. **Solvers**: Ensamble.jl supports various solvers to perform the MIP optimization, ensuring flexibility based on the available resources and licensing preferences.

The system defines test specifications and constraints via a configuration file, then uses a solver to find the optimal item selection.

## Supported Solvers

Ensamble.jl supports several MIP solvers to ensure flexibility for different user needs and environments:

IBM CPLEX: A powerful commercial solver ideal for large-scale, complex test assembly.

CBC (Coin-OR): An open-source solver for smaller-scale applications.

SCIP: Another open-source option suitable for constraint programming.

GLPK: Free and open-source but less efficient for large problems.

HiGHS: A newer open-source solver optimized for high-performance MIP and LP problems.

The user can select the preferred solver in the configuration file.

---

## Features

- **Multiple Form Assembly**: Generate multiple forms simultaneously, ensuring comparability.
- **Flexible Constraints**: Define content constraints, item properties, and test length requirements.
- **Integration with Solvers**: Supports multiple MIP solvers, ensuring flexibility for users based on their environment.
- **Automatic Report Generation**: Generates detailed reports that include item usage, test characteristics, and visualizations (e.g., characteristic curves, information curves).
- **Optimization Algorithms**: Built on robust optimization frameworks, Ensamble.jl efficiently handles large item banks and complex constraint systems.


---

## Usage

### 1. Prepare the Item Bank

The item bank must contain the necessary item parameters. The 3PL IRT model is used by default, but it can be adjusted to use 1, 2 or 3 parameters and approximate a normal distribution model.  Any tool such as **Winsteps**, **ConQuest**, **Bilog**, **Parscale**, or R packages like `TAM` and `MIRT` can be used to calibrate and provide the item parameters.

### 2. Define Test Constraints

Using a configuration file, specify the constraints for your test. This includes:
- Number of forms to be generated
- Minimum and maximum test lengths
- Content balance across areas (e.g., reading, math)
- Use of anchor items

### 3. Run Ensamble.jl

Once the item bank and constraints are prepared, run the test assembly process:

```julia
# Load the configuration and item bank
include("src/ensamble.jl")
using .Ensamble

# Run the assembly process
results = Ensamble.assemble_tests("data/config.toml");
