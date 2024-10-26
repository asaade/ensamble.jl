.
├── data
│  ├── anchor.csv
│  ├── config.toml
│  ├── constraints.csv
│  ├── items.csv
│  ├── model.lp
│  └── solver_config.toml
├── docs
│  ├── structure.md
├── LICENSE
├── Manifest.toml
├── Project.toml
├── README.md
├── results
│  ├── combined_plot.pdf
│  ├── forms.csv
│  ├── model.lp
│  ├── results.csv
│  └── tcc_output.csv
├── src
│  ├── config
│  │  ├── assembly_config_loader.jl
│  │  ├── bank_data_loader.jl
│  │  ├── config_loader.jl
│  │  ├── irt_data_loader.jl
│  │  └── validation.jl
│  ├── configuration.jl
│  ├── constants.jl
│  ├── display
│  │  ├── charts.jl
│  │  └── display_results.jl
│  ├── ensamble.jl
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
└── test_parser.jl
