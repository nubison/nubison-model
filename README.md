# Nubison

This project is a SDK for integrate ML model to Nubison.

## Usage

### Register a model (default)

Logs the experiment and creates a Model Registry entry in one call.

```python
from nubison_model import register, NubisonModel, ModelContext


class MyModel(NubisonModel):
    def load_model(self, context: ModelContext):
        ...

    def infer(self, input):
        ...


model_uri = register(
    MyModel(),
    model_name="MyModel",
    artifact_dirs="src,weights",
    params={"lr": 0.01, "epochs": 100},
    metrics={"accuracy": 0.95},
)
# model_uri == "models:/MyModel/<version>"
```

### Log experiment only (skip Model Registry)

Pass `skip_model_registration=True` to log the experiment and package the
model as a run artifact, without creating a Model Registry entry. The
returned URI has a `runs:/` prefix and can later be registered via
`mlflow.register_model()`.

```python
run_uri = register(
    MyModel(),
    model_name="MyModel",
    artifact_dirs="src,weights",
    params={"lr": 0.01},
    metrics={"accuracy": 0.95},
    skip_model_registration=True,
)
# run_uri == "runs:/<run_id>/model"

# Later, review results and register when ready:
import mlflow

mlflow.register_model(run_uri, "MyModel")
```

The default is `skip_model_registration=False`, which preserves the original
behavior.
