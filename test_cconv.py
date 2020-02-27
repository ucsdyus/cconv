import torch
import torch.nn as nn
import torch.optim as optim
import fastpatch as fp
import cconv


# Graph
# 0: 1, 3
# 1: 0
# 2:
# 3: 0
nn_offset = torch.tensor([0, 2, 3, 3, 4], dtype=torch.int32).cuda()
nn_list = torch.tensor([1, 3, 0, 0], dtype=torch.int32).cuda()
nw_list = torch.tensor(list(range(4 * 3)), dtype=torch.float32).cuda()
grad_nn_offset = torch.tensor([0, 2, 3, 3, 4], dtype=torch.int32).cuda()
grad_nn_list = torch.tensor([1, 0, 3, 0, 0, 0, 0, 1], dtype=torch.int32).cuda()

fp.set_property(2, nn_offset, nn_list, grad_nn_offset, grad_nn_list)
cconv.set_property(3, nw_list)

# Feature 4 x 3 x 1
feat = torch.tensor(
    list(range(12)), dtype=torch.float32).cuda().view(4, 3, 1)

# Model
model = nn.Sequential(
    cconv.CConv(3, 4),
    nn.ReLU(),
    cconv.CConv(4, 1))
model.cuda()


optimizer = optim.SGD(model.parameters(), lr=0.0001)
model.train()
for i in range(10):
    optimizer.zero_grad()

    y = model(feat)
    loss = torch.norm(y)
    print("Step", i, "Loss", loss)

    loss.backward()
    optimizer.step()
