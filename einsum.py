import torch
import numpy as np
"""
Free indices = specified in the output
Summation indices = all other indices
"""

# Transpose
show_transpose = False
X = torch.rand((2, 3))

X_T = torch.einsum("ij -> ji", X)
if show_transpose:
    print("X:", X.shape)
    print("X_T:", X_T.shape)

# Element wise multiplication
show_elem = False
X = torch.rand((3, 2))
Y = torch.rand((3, 2))

elem_mult = torch.einsum("ij,ij -> ij", [X,Y])
if show_elem:
    assert torch.equal(X * Y, elem_mult), f"Got {elem_mult}, expected {X*Y} from X = {X} and Y = {Y}"


# Dot product
show_dot = False
c = torch.rand((3))
v = torch.rand((3))

dot_p = torch.einsum("i,i->",[v,c])

if show_dot:
    assert torch.equal(torch.dot(v, c), dot_p),  f"Got {dot_p}, expected {torch.dot(c, v)} from c = {c} and v = {v}"

# Outer product
show_out = False
c = torch.rand((3))
v = torch.rand((3))

out_p = torch.einsum("i,j->ij",[c,v])
# v = 3x1
# c = 3x1
# want 3x1 * 1x3

if show_out:
    assert torch.equal(torch.outer(c, v), out_p),  f"Got {out_p}, expected {torch.outer(c, v)} from c = {c} and v = {v}"

# Matrix-vector multiplication
show_mat_vec = False
X = torch.rand((3,3))
y = torch.rand((1,3))

out_mat_vec = torch.einsum("ij,kj->ik",[X,y])


if show_mat_vec:
    assert torch.equal(torch.mm(X,y.T), out_mat_vec),  f"Got \n{out_mat_vec}, expected \n{torch.mm(X, y.T)} from X = \n{X} and y = \n{y}"

# Matrix-matrix multiplication
show_mat_mat = True
X = torch.arange(6).reshape((2,3))
Y = torch.arange(12).reshape((3,4))

out_mat_mat = torch.einsum("ij,jk->ik",[X,Y])


if show_mat_mat:
    assert torch.equal(torch.mm(X,Y), out_mat_mat),  f"Got \n{out_mat_mat}, expected \n{torch.mm(X, Y)} from X = \n{X} and Y = \n{Y}"

# Batch matrix multiplication
show_batch_mat_mat = True
X = torch.arange(24).reshape((2,3,4))
Y = torch.arange(40).reshape((2,4,5))

out_mat_mat = torch.einsum("ijk,ikl->ijl",[X,Y])


if show_batch_mat_mat:
    assert torch.equal(torch.bmm(X,Y), out_mat_mat),  f"Got \n{out_mat_mat}, expected \n{torch.bmm(X, Y)} from X = \n{X} and Y = \n{Y}"
