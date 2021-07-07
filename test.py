import torch


score = torch.tensor([[[1,2,7],
                       [4,8,6],
                       [3,5,9]],
                       [[8,1,3],
                        [6,9,4],
                        [2,5,7]]])
print(score)
best_score, best_dst = score.max()
print(best_score,best_dst)
       
