# # nubison-model example

This is an example of how to use the `nubison-model` library to register a user model.

## ## Prerequisites

- mlflow server is running on `http://127.0.0.1:5000` or set `MLFLOW_TRACKING_URI` environment variable.

## ## Example structure

```
example/
├── model.ipynb
├── requirements.txt
└── src/
    └── SimpleLinearModel.py
```

- `model.ipynb`: A notebook file that shows how to register a user model and test it.
- `requirements.txt`: A file that specifies the dependencies of the model.
- `src/`: A directory that contains the source code of the model.

## ## How to register a user model and test it

The `model.ipynb` file shows how to register a user model. It contains the following steps:

1. Define a user model.
2. Register the user model.
3. Test the model.

### ### Define a user model

- The user model should be a class that implements the `NubisonModel` protocol.
- The `load_model` method is used to load the model weights from the file.
- The `infer` method is used to return the inference result.

#### #### `load_model` method

- Use this method to prepare the model for inference which can be time-consuming.
- This method is called once when the model inference server starts.
- The `load_model` method receives a `ModelContext` dictionary containing:
  - `worker_index`: Index of the worker process (0-based) for parallel processing
  - `num_workers`: Total number of workers running the model
- This information is particularly useful for GPU initialization in parallel setups, where you can map specific workers to specific GPU devices.
- The path to the model weights file can be specified relative.

#### #### `infer` method

- This method is called for each inference request.
- This method can take any number of arguments.
- The return and argument types of the `infer` method can be `int`, `float`, `str`, `list`(`Tensor`, `ndarray`) and `dict`.

### ### Register a user model

- The `register` function is used to register the user model.
- The `artifact_dirs` argument specifies the folders containing the files used by the model class.
- If the model class does not use any files, this argument can be omitted.

### ### Test a user model

- The `test_client` function is used to test the model.
- It can be used to test the model through HTTP requests.

## ## requirements.txt

- The `requirements.txt` file is used to specify the dependencies of the model.
- The packages listed here will be installed in the environment where the model is deployed.
- If no `requirements.txt` file is provided, current environment packages will be used.

## ## src

- The `src/` directory contains the source code of the model.
- The name `src/` can be changed to any other name and additional folders can be added. The folders should be specified in the `artifact_dirs` argument of the `register` function.
- The files in those folders should be imported using absolute paths from `model.ipynb`.
- Both relative and absolute paths can be used when importing inside the `src/` directory. Check the `SimpleLinearModel.py` and `utils/logger.py` for more details.
