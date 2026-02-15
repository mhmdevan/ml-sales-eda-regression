from locust import HttpUser, between, task


class SalesApiUser(HttpUser):
    wait_time = between(0.1, 0.3)

    @task
    def predict_sales(self) -> None:
        self.client.post(
            "/predict",
            json={
                "QUANTITYORDERED": 25,
                "PRICEEACH": 80.0,
                "ORDERLINENUMBER": 2,
                "MSRP": 104.0,
                "QTR_ID": 3,
                "MONTH_ID": 7,
                "YEAR_ID": 2004,
                "PRODUCTLINE": "Classic Cars",
                "COUNTRY": "USA",
                "DEALSIZE": "Small",
            },
            timeout=10,
        )
