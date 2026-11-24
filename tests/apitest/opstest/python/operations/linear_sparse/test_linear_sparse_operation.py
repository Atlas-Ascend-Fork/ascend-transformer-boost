#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import os
import sys
import json
import unittest
import subprocess
import builtins
import torch
import torch_npu
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

OP_NAME = "LinearSparseOperation"
PARAM = {"transposeA": False, "transposeB": True, "tilingK": 8, "tilingN": 8}


class TestLinearSparseOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        dim0 = in_tensors[0].size()[0]
        dim1 = in_tensors[0].size()[1]
        out_tensor = torch.rand(dim0, dim1).npu().half()
        return [out_tensor]

    def golden_compare(self, out_tensor, golden_out_tensor):
        print("out_tensor.shape", out_tensor.shape)
        print("out_tensor:", out_tensor)
        return True

    def test(self):
        if not operation_test.get_soc_version() == 'Ascend310P':
            print("this testcase only supports Ascend310p")
            return

        input0 = torch.randint(low=-5, high=5, size=(1, 4096), dtype=torch.int8).npu()
        weight_npu = torch.randint(low=-5, high=5, size=(4096, 4096), dtype=torch.int8).npu()
        input2 = torch.randint(low=-10, high=10, size=(4096,), dtype=torch.int).npu()
        input3 = torch.randint(low=-2, high=2, size=(4096,), dtype=torch.int).float()
        input3 = np.frombuffer(input3.numpy().tobytes(), dtype=np.uint32).astype(np.int64)
        input3 = torch.tensor(input3, dtype=torch.int64).npu()

        try:
            import msmodelslim
            from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor

            package_root = os.path.dirname(os.path.abspath(msmodelslim.__file__))
            compress_graph_path = os.path.join(package_root, "pytorch", "weight_compression", "compress_graph")
        except ImportError:
            ascend_home_path = os.getenv("ASCEND_HOME_PATH", "")
            legacy_path = os.path.join(ascend_home_path, "tools", "msmodelslim", "pytorch")
            sys.path.append(legacy_path)
            from weight_compression import CompressConfig, Compressor

            compress_graph_path = os.path.join(legacy_path, "weight_compression", "compress_graph")

        compile_result_path = os.path.join(compress_graph_path, "build", "compress_excutor")
        if not os.path.exists(compile_result_path):
            result = subprocess.run(
                ["bash", "build.sh", os.getenv("ASCEND_HOME_PATH", "")],
                cwd=compress_graph_path,
                check=False,
            )
            if result.returncode != 0 or not os.path.exists(compile_result_path):
                raise RuntimeError("compile compress_weight tool failed")

        work_dir = os.path.join(os.getcwd(), "linear_sparse_weight")
        os.makedirs(work_dir, mode=0o700, exist_ok=True)
        os.chmod(work_dir, 0o700)

        def compress(weight, filename):
            weight_path = os.path.join(work_dir, filename)
            np.save(weight_path, {"key": weight.cpu()})
            os.chmod(weight_path, 0o600)
            compress_config = CompressConfig(record_detail_root=work_dir)
            original_input = builtins.input

            def input_yes(prompt=""):
                if prompt == "Please enter 'yes' or 'no'\n":
                    return "yes"
                return original_input(prompt)

            builtins.input = input_yes
            try:
                compressor = Compressor(compress_config, weight_path)
                return compressor.run()
            finally:
                builtins.input = original_input

        compress_weight, compress_idx, _ = compress(weight_npu, "weight_nd.npy")
        input1 = torch.tensor(compress_weight["key"].astype(np.int8)).npu()
        input4 = torch.tensor(compress_idx["key"].astype(np.int8)).npu()
        self.execute(OP_NAME, json.dumps(PARAM), [input0, input1, input2, input3, input4])

        weight_npu = torch_npu.npu_format_cast(weight_npu, 29)
        compress_weight, compress_idx, _ = compress(weight_npu, "weight_nz.npy")
        input1 = torch.tensor(compress_weight["key"].astype(np.int8)).npu()
        input4 = torch.tensor(compress_idx["key"].astype(np.int8)).npu()
        self.execute(OP_NAME, json.dumps(PARAM), [input0, input1, input2, input3, input4])


if __name__ == '__main__':
    unittest.main()
