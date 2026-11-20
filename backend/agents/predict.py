from ml.inference.predict import ChurnPredictor


class PredictAgent:
    def __init__(self, predictor: ChurnPredictor | None = None): self.predictor = predictor or ChurnPredictor()
    def run(self, customer_id: str, features: dict) -> dict:
        return self.predictor.predict(customer_id, features)
