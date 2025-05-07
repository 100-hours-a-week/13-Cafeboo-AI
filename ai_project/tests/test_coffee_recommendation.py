import unittest
from ai_project.models.coffee_recommendation import CoffeeRecommendation

class TestCoffeeRecommendation(unittest.TestCase):

    def setUp(self):
        self.recommender = CoffeeRecommendation()

    def test_recommend(self):
        user_profile = {
            "caffeine_sensitivity": 70,
            "preferred_flavor": "bitter"
        }
        recommendations = self.recommender.recommend(user_profile)
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

if __name__ == "__main__":
    unittest.main()
