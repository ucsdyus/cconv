import torch
import fastpatch as fp


# Dynamic Graph
# Graph
# 0: 1, 3
# 1: 0
# 2:
# 3: 0
nn_offset = torch.tensor([0, 2, 3, 3, 4], dtype=torch.int32).cuda()
nn_list = torch.tensor([1, 3, 0, 0], dtype=torch.int32).cuda()
grad_nn_offset = torch.tensor([0, 2, 3, 3, 4], dtype=torch.int32).cuda()
grad_nn_list = torch.tensor([1, 0, 3, 0, 0, 0, 0, 1], dtype=torch.int32).cuda()

# 4 x 3
feat = torch.tensor(
    list(range(12)), dtype=torch.float32, requires_grad=True).cuda().view(4, 3, 1)
feat.retain_grad()
weight = torch.tensor(
    list(range(3)), dtype=torch.float32, requires_grad=True).view(1, 1, 1, 3).cuda()
weight.retain_grad()

fp.update_feat_config(2, nn_offset, nn_list, grad_nn_offset, grad_nn_list)
patchfeat = fp.feat_patch(feat)  # 4 x 2 x 3 x 1
patchfeat.retain_grad()
print("patchfeat shape", patchfeat.size())

y = torch.matmul(weight, patchfeat).sum(axis=1)  # 4 x 1 x 1
y.retain_grad()
print("result shape", y.size())

loss = torch.norm(y)
loss.retain_grad()
print("loss", loss)
loss.backward()

print("feat grad")
for i in range(4):
    print(i, feat.grad[i])

# Selection Mat
nw_list = torch.tensor(list(range(4 * 3)), dtype=torch.float32).cuda()  # spatial = 3
# 4 x 2 x 3 x 1
select_mat = fp.selection_mat_patch(nn_offset, nw_list, 2, 3)
print("Select Mat", select_mat.size())
for i in range(4):
    print(i, select_mat[i])

# Fixed Points
# Graph
# 0: 0 1
# 1: 0
# 2: 1
# 3:
# 4: 0 1
nn_offset = torch.tensor([0, 2, 3, 4, 4, 6], dtype=torch.int32).cuda()
nn_list = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int32).cuda()

fixed_in = torch.tensor(
    list(range(6)), dtype=torch.float32, requires_grad=False).cuda().view(2, 3, 1)
weight = torch.tensor(
    list(range(3)), dtype=torch.float32, requires_grad=True).view(1, 1, 1, 3).cuda()
weight.retain_grad()

fp.update_fixed_config(2, nn_offset, nn_list)
patch_fixed = fp.fixed_patch(fixed_in)  # 5 x 2 x 3 x 1

y = torch.matmul(weight, patch_fixed).sum(axis=1)  # 5 x 1 x 1
y.retain_grad()
print("result shape", y.size())

loss = torch.norm(y)
loss.retain_grad()
print("loss", loss)
loss.backward()


print("weight grad shape", weight.grad.size())
print("patch_fixed shape", patch_fixed.size())
