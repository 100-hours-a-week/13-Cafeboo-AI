import unittest
from fastapi.testclient import TestClient
from ai_project.main import app

class TestAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_caffeine_limit(self):
        response = self.client.post("/predict", json={
            "gender": "M",
            "age": 30,
            "height": 175,
            "weight": 70,
            "sleep_duration": 7,
            "sleep_quality": "good",
            "caffeine_sensitivity": 50,
            "total_caffeine_today": 200
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("caffeine_limit", response.json())

if __name__ == "__main__":
    unittest.main()
