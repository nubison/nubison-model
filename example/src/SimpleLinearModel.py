# Modules under artifact folders can be imported using both relative or absolute paths.
from src.utils.logger import get_logger  # type: ignore


class SimpleLinearModel:
    """
    A simple linear model.
    """

    def __init__(self, weight_path):
        """
        Initialize the model.
        """
        self.logger = get_logger("SimpleLinearModel")
        self.load_weight(weight_path)

    def load_weight(self, weight_path):
        """
        Load the weights from the file.
        """
        try:
            with open(weight_path, "r") as file:
                self.weights = [float(w) for w in file.read().split()]
            self.logger.info(f"Weights loaded successfully from {weight_path}.")
        except FileNotFoundError:
            self.logger.error(f"The file at {weight_path} was not found.")
            raise

    def calc(self, x1: float, x2: float):
        """
        Calculate the result of the linear model.
        """
        self.logger.info(
            f"Calculating the result of the linear model with x1={x1}, x2={x2}."
        )
        return self.weights[0] * x1 + self.weights[1] * x2 + self.weights[2]
