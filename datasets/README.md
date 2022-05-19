## Dataset download
Each real-world use case requires a specific dataset to reproduce the experiment.
Datasets are stored under `datasets/<dataset name>` for the code to find it.
If you decide to use other datasets, change the correponding variables in
`data_nilm.py` and `data_paf.py`.

### Disaggregation test case (NILM)
- UK-DALE (Kelly2015):
    - Link: https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated
    - Path: `datasets/UK-DALE/ukdale2017.h5`

- REFIT (Murray2017):
    - Link: https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned
    - Path: `datasets/REFIT/refit.h5`

### Atrial fibrillation test case (PAF)
- Physionet-PAF(Moody2001):
    - Link: https://physionet.org/content/afpdb/1.0.0/
    - Path to extract files: `datasets/Physionet-PAF/`
