import functools
import random

import numpy as np

from harness.seeding import find_stray_generators, reseed


class Sampler:
    def __init__(self, rng):
        self.rng = rng


class FakeRobosuiteEnv:
    def __init__(self):
        self.rng = np.random.default_rng(999)
        self.placement = Sampler(self.rng)  # shares the generator by identity


def test_reseed_resets_the_shared_generator_in_place():
    env = FakeRobosuiteEnv()
    rng = env.rng
    env.rng.random(5)
    reseed(env, 42)
    first = (env.placement.rng.random(3), np.random.random(), random.random())
    reseed(env, 42)
    second = (env.placement.rng.random(3), np.random.random(), random.random())
    assert env.rng is rng and env.placement.rng is rng
    assert np.array_equal(first[0], second[0]) and first[1:] == second[1:]
    assert np.array_equal(first[0], np.random.default_rng(42).random(3))
    reseed(env, 43)
    assert not np.array_equal(env.rng.random(3), first[0])


class Slotted:
    __slots__ = ("inner",)

    def __init__(self, inner):
        self.inner = inner


def test_walker_reports_every_stray_generator_by_path():
    robo = FakeRobosuiteEnv()
    stray_closure = np.random.default_rng(1)
    env = Sampler(None)
    env.robosuite_env = robo
    env._rng = np.random.default_rng(0)
    env.nested = {"list": [1, "x", Slotted(np.random.RandomState(0))]}
    env.factory = lambda: stray_closure
    env.partial = functools.partial(print, np.random.default_rng(2))
    env.method = robo.placement.__init__  # bound method -> its __self__
    env.cycle = env
    env.array = np.zeros(3)
    found = find_stray_generators(env, robo.rng, root_name="env")
    assert sorted(found) == sorted(
        [
            "env._rng",
            "env.nested['list'][2].inner",
            "env.factory.<closure>",
            "env.partial.args[0]",
        ]
    )


def test_walker_accepts_a_clean_graph():
    robo = FakeRobosuiteEnv()
    env = Sampler(robo.rng)
    env.robosuite_env = robo
    env.samplers = (robo.placement, {robo.placement})
    assert find_stray_generators(env, robo.rng) == []
