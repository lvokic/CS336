from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

setup(
    packages=[],
    ext_modules=[
        Pybind11Extension(
            "_cs336_bpe", ["src/bindings.cpp"],
            depends=["src/training.hpp", "src/pair_queue.hpp"],
            cxx_std=17,
            extra_compile_args=["-O3"],
            extra_link_args=["-pthread"],
        )
    ],
    cmdclass={"build_ext": build_ext},
)
