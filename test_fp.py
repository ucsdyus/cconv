import torch
import fastpatch as fp


# Graph
# 0: 1, 3
# 1: 0
# 2:
# 3: 0
nn_offset = torch.tensor([0, 2, 3, 3, 4], dtype=torch.int32).cuda()
nn_list = torch.tensor([1, 3, 0, 0], dtype=torch.int32).cuda()
grad_nn_offset = torch.tensor([0, 2, 3, 3, 4], dtype=torch.int32).cuda()
grad_nn_list = torch.tensor([1, 0, 3, 0, 0, 0, 0, 1], dtype=torch.int32).cuda()

fp.set_property(2, nn_offset, nn_list, grad_nn_offset, grad_nn_list)

# 4 x 3
feat = torch.tensor(
    list(range(12)), dtype=torch.float32, requires_grad=True).cuda().view(4, 3, 1)
feat.retain_grad()
weight = torch.tensor(
    list(range(3)), dtype=torch.float32, requires_grad=True).view(1, 1, 1, 3).cuda()
weight.retain_grad()

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

nw_list = torch.tensor(list(range(4 * 3)), dtype=torch.float32).cuda()  # spatial = 3
select_mat = fp.selection_mat_patch(nw_list, 3)
print("Select Mat", select_mat.size())
for i in range(4):
    print(i, select_mat[i])