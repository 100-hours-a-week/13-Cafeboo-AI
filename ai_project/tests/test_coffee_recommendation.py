import unittest
from ai_project.models.coffee_recommendation import CoffeeRecommendation

class TestCoffeeRecommendation(unittest.TestCase):

    def setUp(self):
        self.recommender = CoffeeRecommendation()

    def test_recommend(self):
        user_profile = {
            "gender": 1,
            "age": 27,
            "height": 165.0,
            "weight": 45.1,
            "is_smoker": 1,
            "take_hormonal_contraceptive": 0,
            "caffeine_sensitivity": 90,
            "current_caffeine": 340,
            "caffeine_limit": 400,
            "residual_at_sleep": 38.7,
            "target_residual_at_sleep": 50.0,
            "planned_caffeine_intake": 125,
            "current_time": 15.5,
            "sleep_time": 17.5
        }
        recommendations = self.recommender.recommend(user_profile)
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

if __name__ == "__main__":
    unittest.main()
