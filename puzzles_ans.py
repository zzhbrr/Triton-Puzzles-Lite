import argparse
from typing import List

import triton
import triton.language as tl


# Local imports
from display import print_end_line
from tensor_type import Float32, Int32
from test_puzzle import test

r"""
## Puzzle 1: Constant Add

Add a constant to a vector. Uses one program id axis. 
Block size `B0` is always the same as vector `x` with length `N0`.

.. math::
    z_i = 10 + x_i \text{ for } i = 1\ldots N_0
"""

def add_spec(x: Float32[32,]) -> Float32[32,]:
    "This is the spec that you should implement. Uses typing to define sizes."
    return x + 10.

@triton.jit
def add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    range_x = tl.arange(0, B0)
    x = tl.load(x_ptr + range_x)
    # Finish me!
    x = x + 10.
    tl.store(z_ptr + range_x, x)


r"""
## Puzzle 2: Constant Add Block

Add a constant to a vector. Uses one program block axis (no `for` loops yet). 
Block size `B0` is now smaller than the shape vector `x` which is `N0`.

.. math::
    z_i = 10 + x_i \text{ for } i = 1\ldots N_0
"""

def add2_spec(x: Float32[200,]) -> Float32[200,]:
    return x + 10.

@triton.jit
def add_mask2_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    # Finish me!
    block_id = tl.program_id(0)
    range_x = block_id * B0 + tl.arange(0, B0)
    mask = range_x < N0
    x = tl.load(x_ptr + range_x, mask=mask)
    x = x + 10.
    tl.store(z_ptr + range_x, x, mask=mask)
    return


r"""
## Puzzle 3: Outer Vector Add

Add two vectors.

Uses one program block axis. Block size `B0` is always the same as vector `x` length `N0`.
Block size `B1` is always the same as vector `y` length `N1`.

.. math::
    z_{j, i} = x_i + y_j\text{ for } i = 1\ldots B_0,\ j = 1\ldots B_1
"""

def add_vec_spec(x: Float32[32,], y: Float32[32,]) -> Float32[32, 32]:
    return x[None, :] + y[:, None]

@triton.jit
def add_vec_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    # Finish me!
    range_x = tl.arange(0, B0)
    range_y = tl.arange(0, B1)
    range_z = range_y[:, None] * B0 + range_x[None, :]
    # print(range_z)
    x = tl.load(x_ptr + range_x)
    y = tl.load(y_ptr + range_y)
    z = y[:, None] + x[None, :]
    # print(z)
    tl.store(z_ptr + range_z, z)
    return


r"""
## Puzzle 4: Outer Vector Add Block

Add a row vector to a column vector.

Uses two program block axes. Block size `B0` is always less than the vector `x` length `N0`.
Block size `B1` is always less than vector `y` length `N1`.

.. math::
    z_{j, i} = x_i + y_j\text{ for } i = 1\ldots N_0,\ j = 1\ldots N_1
"""

def add_vec_block_spec(x: Float32[100,], y: Float32[90,]) -> Float32[90, 100]:
    return x[None, :] + y[:, None]

@triton.jit
def add_vec_block_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    return

def run_puzzles(args, puzzles: List[int]):
    print_log = args.log

    if 1 in puzzles:
        print("Puzzle #1:")
        ok = test(add_kernel, add_spec, nelem={"N0": 32}, print_log=print_log)
        print_end_line()
        if not ok:
            return
    if 2 in puzzles:
        print("Puzzle #2:")
        ok = test(add_mask2_kernel, add2_spec, nelem={"N0": 200}, print_log=print_log)
        print_end_line()
        if not ok:
            return
    if 3 in puzzles:
        print("Puzzle #3:")
        ok = test(add_vec_kernel, add_vec_spec, nelem={"N0": 32, "N1": 32}, print_log=print_log)
        print_end_line()
        if not ok:
            return
    if 4 in puzzles:
        print("Puzzle #4:")
        ok = test(add_vec_block_kernel, add_vec_block_spec, nelem={"N0": 100, "N1": 90}, print_log=print_log)
        print_end_line()
        if not ok:
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--puzzle", type=int, metavar="N", help="Run Puzzle #N")
    parser.add_argument("-a", "--all", action="store_true", help="Run all Puzzles. Stop at first failure.")
    parser.add_argument("-l", "--log", action="store_true", help="Print log messages.")
    args = parser.parse_args()

    if args.all:
        run_puzzles(args, list(range(1, 10)))
    elif args.puzzle:
        run_puzzles(args, [int(args.puzzle)])
    else:
        parser.print_help()

