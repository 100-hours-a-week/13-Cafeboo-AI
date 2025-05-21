import unittest
from ai_project.models.caffeine_limit import CaffeineLimitModel

class TestCaffeineLimitModel(unittest.TestCase):

    def setUp(self):
        """테스트 전에 실행되는 초기 설정"""
        self.model = CaffeineLimitModel()

    def test_predict_valid_input(self):
        """유효한 입력으로 모델 예측 테스트"""
        test_input = {
            "gender": 1,
            "age": 30,
            "height": 175,
            "weight": 70,
            "sleep_duration": 7,
            "sleep_quality": "good",
            "caffeine_sensitivity": 50,
            "total_caffeine_today": 200
        }
        result = self.model.predict(test_input)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

    def test_predict_invalid_input(self):
        """잘못된 입력으로 모델 예측 테스트"""
        with self.assertRaises(ValueError):
            self.model.predict(None)

if __name__ == "__main__":
    unittest.main()
