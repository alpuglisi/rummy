import os
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

ext_modules = [
    Pybind11Extension(
        "rummy_engine",
        [os.path.join(THIS_DIR, "rummy_env.cpp")],
        cxx_std=17,
        extra_compile_args=["-O3", "-ffast-math", "-march=native"]
    ),
]

setup(
    name="rummy_engine",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
