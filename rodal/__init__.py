# -*- coding: utf-8 -*-
from gym.envs.registration import register

register(
    id='rodal-v0',
    entry_point='rodal.envs:PtDiscreto6x5',
)
