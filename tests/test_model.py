import unittest
from src.model import WineQualityModel

class TestWineQualityModel(unittest.TestCase):
    def test_train_and_predict(self):

        # Minimal mock data
        X = [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4],
             [7.8, 0.88, 0.0, 2.6, 0.098, 25.0, 67.0, 0.9968, 3.2, 0.68, 9.8]]
        y = [5, 5]

        model = WineQualityModel()
        model.train(X, y)
        preds = model.predict(X)

        self.assertEqual(list(preds), y)

if __name__ == '__main__':

    unittest.main()
