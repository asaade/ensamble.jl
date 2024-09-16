.
├── data
│  ├── anchor.csv
│  ├── config.toml
│  ├── constraints.csv
│  ├── items.csv
│  └── solver_config.toml
├── docs
│  └── structure.md
├── LICENSE
├── Manifest.toml
├── Project.toml
├── README.md
├── results
│  ├── combined_plot.pdf
│  ├── model.lp
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
