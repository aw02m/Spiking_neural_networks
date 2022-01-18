import numpy as np
from ..network import KNNetParameters, KNNet, KNNetState
from ..method import ForceParameters, ForceLearn

def test_force_create_object():
    net = KNNet(1, 1, 1)
    ForceLearn(net=net)

def test_force_create_object_with_self_parameters():
    lp = ForceParameters(lr = .213)
    net = KNNet(1, 1, 1)
    fl = ForceLearn(net=net, lp=lp)
    assert fl.params.lr == lp.lr

def test_force_train():
    lp = ForceParameters(lr = .213)
    net = KNNet(1, 1, 1)
    fl = ForceLearn(net=net, lp=lp)
    fl.train(target_outputs=np.array([[1, 1, 1, 1, 1]]).reshape(5, 1))
