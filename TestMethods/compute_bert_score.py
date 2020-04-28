from bert_score import score
import matplotlib.pyplot as plt
import sys



def main(refs_path, preds_path, model_name):
   with open("../Experiments/" + modelName +"/Refs.txt", 'r') as file:
       refs = [line.strip() for line in file]
   with open("../Experiments/" + modelName +"/Preds.txt", 'r') as file:
       preds = [line.strip() for line in file]
   P, R, F1 = score(preds, refs, lang="en", verbose=True)
   print(f"System level P score: {P.mean():.3f}")
   print(f"System level R score: {R.mean():.3f}")
   print(f"System level F1 score: {F1.mean():.3f}")
   with open("../Experiments/" + modelName +"/bert_scores.txt", 'a+') as file:
       file.write("System level P score: {P.mean():.3f}\n" +
                  "System level R score: {R.mean():.3f}\n" +
                  "System level F1 score: {F1.mean():.3f}")
   plt.hist(F1, bins=20)
   plt.savefig('../Experiments/' + model_name + '/F1_bert_score_histogram.png')
   plt.hist(P, bins=20)
   plt.savefig('../Experiments/' + model_name + '/P_bert_score_histogram.png')
   plt.hist(R, bins=20)
   plt.savefig('../Experiments/' + model_name + '/R_bert_score_histogram.png')




if __name__ == "__main__":
  args = sys.argv
  refs_path = args[1]
  preds_path = args[2]
  model_name = args[3]
  main(refs_path, preds_path, model_name)
