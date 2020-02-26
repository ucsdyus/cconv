import torch
import fastpatch_impl

# Graph
# 0: 1, 3
# 1: 0
# 2:
# 3: 0
nn_offset = torch.tensor([0, 2, 3, 3, 4], dtype=torch.int32).cuda()
nn_list = torch.tensor([1, 3, 0, 0], dtype=torch.int32).cuda()
feat = torch.tensor(list(range(12)), dtype=torch.float32).cuda().view(4, 3)

patchfeat = fastpatch_impl.feat_forward(feat, nn_offset, nn_list, 2)  # 4 x 2 x 3 x 1

print("Feature")
print(feat.size())
for i in range(4):
    print(i, feat[i])


print("Patch Feature")
print(patchfeat.size())
for i in range(4):
    print(i, patchfeat[i])


# Grad Graph
# 0: 1, 3
# 1: 0
# 2:
# 3: 0
# Grad v offset
# 0: 0, 0
# 1: 0
# 2:
# 3: 1
grad_nn_offset = torch.tensor([0, 2, 3, 3, 4], dtype=torch.int32).cuda()
grad_nn_list = torch.tensor([1, 0, 3, 0, 0, 0, 0, 1], dtype=torch.int32).cuda()
grad_patchfeat = torch.tensor(
    list(range(24)), dtype=torch.float32).cuda().view(4, 2, 3, 1)  # 4 x 2 x 3 x 1

grad_feat = fastpatch_impl.feat_backward(grad_patchfeat, grad_nn_offset, grad_nn_list, 2)  # 4 x 3 x 1

print("Grad Patch Feature")
print(grad_patchfeat.size())
for i in range(4):
    print(i, grad_patchfeat[i])


print("Grad Feature")
print(grad_feat.size())
for i in range(4):
    print(i, grad_feat[i])
