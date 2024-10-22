from mlflow.tracking import MlflowClient
from nubison_model.Model import Model, register


def test_register_model(mlflow_server):
    """
    Test registering a model to MLflow's Model Registry.
    """
    model_name = "TestRegisteredModel"

    # Define a simple model (for example purposes, using a dummy model)
    class DummyModel(Model):
        pass

    # Register the model
    register(DummyModel(), model_name=model_name, mlflow_uri=mlflow_server)

    # Retrieve the registered model
    client = MlflowClient()
    registered_model = client.get_registered_model(model_name)

    # Assertions to verify the model registration
    assert registered_model.name == model_name

    # Cleanup: Delete the registered model after the test
    client.delete_registered_model(model_name)
