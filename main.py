from src.inference import SentimentPredictor





if __name__ == '__main__':
    predictor = SentimentPredictor()
    print(predictor.predict("This movie was absolutely bad!"))
