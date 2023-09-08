#!/usr/bin/env python
import web
import trainer
import json


def classify(vectorizer, classifier, data):
    labels = ['cpp', 'java', 'javascript', 'python', 'scala', 'text']
    vectorized_data = vectorizer.transform([data])
    prediction = classifier.predict(vectorized_data[0])
    print("prediction: %s" % prediction)
    score_index = labels.index(prediction)
    score = classifier.predict_proba(vectorized_data[0])
    print("score %s" % score)
    score_percentage = round(score[0, score_index] * 100)
    return prediction[0], score_percentage


class Predict:

    def POST(self):
        print("in predict")
        # Get the input data from the POST request
        data = web.data().decode("utf-8")
        input_data = json.loads(data)
        web.header('Content-Type', 'application/json; charset=utf-8')
        web.header('Access-Control-Allow-Origin', '*')

        if "query" not in input_data:
            return web.badrequest()

        # Extract the query from the input data
        query = input_data["query"]
        print(f"Query: {query}")
        classification, score = classify(vectorizer, classifier, query)
        if len(classification) == 0:
            return {"classification": "NONE"}
        return json.dumps({'classification': classification, 'score': score})

if __name__ == "__main__":
    accept = '*/*'
    content_type = 'application/json'
    urls = ("/predict", "Predict")
    web_app = web.application(urls, globals())
    vectorizer, classifier = trainer.train()
    web_app.run()

