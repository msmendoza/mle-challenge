# Challenge

- Normally I do not version any `data/` folder / files, but for the sake of this challenge I will include the `data/` folder in the repository.
- I use `poetry` to manage package dependencies and virtual environments, because of cleaner deps listings, integrations with linterns and formatters and deps separation between environments. For the sake of this challenge I will not use it.
- I use IaC (Terraform) to manage **all** cloud resources, but for the sake of this challenge I will not use it. What
would change in that case is the direct-deployment in this repo. That would be done in another centralized infra
repo and this repo only updates associated Docker images.
- I use Artifact Registry to store images given an IaC approach, this is skipped in this challenge.

## `challenge/exploration.ipynb` fixes

1. `training_data` var was not being used.
2. Although some better cols where found after some improvements I left the ones fixed in the
tests.
3. I chosed the Logistic Regression model because it is a simple model in comparison to the
XGBoost model and performances are pretty similar > low processing costs.
