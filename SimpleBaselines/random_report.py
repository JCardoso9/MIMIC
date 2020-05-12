import sys
sys.path.append('../Utils/')

import json
from tqdm import tqdm
from generalUtilities import *
from nlgeval import NLGEval

import random

def get_random_report(study_id_list, train_reports):
    random_index = random.randint(0, len(study_id_list)-1)
    random_report = train_reports[study_id_list[random_index]]
    return unifyCaption(random_report)


def main():

    references = [[]]
    hypotheses = []

   # Create NlG metrics evaluator
    nlgeval = NLGEval(metrics_to_omit=['SkipThoughtCS', 'GreedyMatchingScore', 'VectorExtremaCosineSimilarity', 'EmbeddingAverageCosineSimilarity'])

    with open('/home/jcardoso/MIMIC/encodedTestCaptionsF.json') as json_file:
        referenceCaptionsDict = json.load(json_file)

    with open('/home/jcardoso/MIMIC/encodedTrainCaptionsF.json') as json_file:
        KBCaptionsDict = json.load(json_file)

    reference_ids = list(referenceCaptionsDict.keys())

    KB_ids = list(KBCaptionsDict.keys())

    for i in tqdm(range(len(referenceCaptionsDict.keys()))):
        references[0].append(unifyCaption(referenceCaptionsDict[reference_ids[i]]))
        hypotheses.append(get_random_report(KB_ids, KBCaptionsDict))

    metrics_dict = nlgeval.compute_metrics(references, hypotheses)
    print(metrics_dict)

    with open("random_TestResults.txt", "w+") as file:
      for metric in metrics_dict:
        file.write(metric + ":" + str(metrics_dict[metric]) + "\n")


if __name__ == "__main__":
    main()
