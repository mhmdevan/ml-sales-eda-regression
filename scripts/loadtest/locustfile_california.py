from locust import HttpUser, between, task


class CaliforniaApiUser(HttpUser):
    wait_time = between(0.1, 0.3)

    @task
    def predict_california(self) -> None:
        self.client.post(
            "/predict",
            json={
                "MedInc": 8.0,
                "HouseAge": 22,
                "AveRooms": 5.5,
                "AveBedrms": 1.1,
                "Population": 700,
                "AveOccup": 2.1,
                "Latitude": 37.7,
                "Longitude": -122.4,
            },
            timeout=10,
        )
