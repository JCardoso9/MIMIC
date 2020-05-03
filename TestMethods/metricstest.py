from nlgeval import NLGEval


# Create NlG metrics evaluator
nlgeval = NLGEval(metrics_to_omit=['SkipThoughtCS', 'GreedyMatchingScore', 'VectorExtremaCosineSimilarity', 'EmbeddingAverageCosineSimilarity'])



references = [['hello this is tim from the office.', 'on a plate i step as i roam from asgard.']]

predictions = ['hello this is tim from the office', 'as i roam from asgard i step on a plate.']

metrics_dict = nlgeval.compute_metrics(references, predictions)




print(metrics_dict)
