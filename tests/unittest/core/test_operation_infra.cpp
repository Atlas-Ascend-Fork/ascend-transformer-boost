/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include "atb/operation_infra.h"

namespace {
class TestContext final : public atb::Context {
public:
    TestContext()
    {
        executeStreams_ = {reinterpret_cast<aclrtStream>(&streamStorage_[0]),
                           reinterpret_cast<aclrtStream>(&streamStorage_[1])};
    }

    atb::Status SetExecuteStream(aclrtStream stream) override
    {
        executeStreams_.at(0) = stream;
        return atb::NO_ERROR;
    }
    aclrtStream GetExecuteStream() const override
    {
        return executeStreams_.at(0);
    }
    atb::Status SetAsyncTilingCopyStatus(bool) override
    {
        return atb::NO_ERROR;
    }
    bool GetAsyncTilingCopyStatus() const override
    {
        return false;
    }
    atb::Status SetExecuteStreams(const std::vector<aclrtStream> &streams) override
    {
        executeStreams_ = streams;
        return atb::NO_ERROR;
    }
    std::vector<aclrtStream> GetExecuteStreams() override
    {
        return executeStreams_;
    }
    atb::Status SetExecuteType(atb::ExecuteType) override
    {
        return atb::NO_ERROR;
    }
    atb::ExecuteType GetExecuteType() override
    {
        return atb::EXECUTE_NORMAL;
    }
    atb::Status SetLaunchMode(atb::LaunchMode) override
    {
        return atb::NO_ERROR;
    }
    atb::LaunchMode GetLaunchMode() override
    {
        return atb::KERNEL_LAUNCH_MODE;
    }

private:
    int streamStorage_[2] = {};
    std::vector<aclrtStream> executeStreams_;
};

class TestOperation final : public atb::OperationInfra {
public:
    std::string GetName() const override
    {
        return "TestOperationInfra";
    }
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &, atb::SVector<atb::TensorDesc> &) const override
    {
        return atb::NO_ERROR;
    }
    uint32_t GetInputNum() const override
    {
        return 0;
    }
    uint32_t GetOutputNum() const override
    {
        return 0;
    }
    atb::Status Setup(const atb::VariantPack &, uint64_t &, atb::Context *) override
    {
        return atb::NO_ERROR;
    }
    atb::Status Execute(const atb::VariantPack &, uint8_t *, uint64_t, atb::Context *) override
    {
        return atb::NO_ERROR;
    }
};
} // namespace

TEST(TestOperationInfra, CopyAssignmentCopiesStreamId)
{
    TestContext context;
    TestOperation source;
    TestOperation copy;
    source.SetExecuteStreamId(1);

    copy = source;

    EXPECT_EQ(copy.GetExecuteStreamId(), 1);
    EXPECT_EQ(copy.GetExecuteStream(&context), context.GetExecuteStreams().at(1));
}

TEST(TestOperationInfra, SelfAssignmentPreservesStreamIdAndSelection)
{
    TestContext context;
    TestOperation operation;
    operation.SetExecuteStreamId(1);
    const auto selectedStream = operation.GetExecuteStream(&context);

    operation = operation;

    EXPECT_EQ(operation.GetExecuteStreamId(), 1);
    EXPECT_EQ(operation.GetExecuteStream(&context), selectedStream);
}

TEST(TestOperationInfra, SelfAssignmentPreservesSelectionAcrossStreams)
{
    TestContext context;
    TestOperation streamZeroOperation;
    TestOperation streamOneOperation;
    streamZeroOperation.SetExecuteStreamId(0);
    streamOneOperation.SetExecuteStreamId(1);

    const std::vector<aclrtStream> selectedStreamsBefore = {streamZeroOperation.GetExecuteStream(&context),
                                                            streamOneOperation.GetExecuteStream(&context)};
    ASSERT_NE(selectedStreamsBefore.at(0), selectedStreamsBefore.at(1));

    streamOneOperation = streamOneOperation;

    const std::vector<aclrtStream> selectedStreamsAfter = {streamZeroOperation.GetExecuteStream(&context),
                                                           streamOneOperation.GetExecuteStream(&context)};
    EXPECT_EQ(selectedStreamsAfter, selectedStreamsBefore);
}

TEST(TestOperationInfra, GetExecuteStreamRejectsNullContext)
{
    TestOperation operation;

    EXPECT_EQ(operation.GetExecuteStream(nullptr), nullptr);
}

TEST(TestOperationInfra, GetExecuteStreamRejectsOutOfRangeStreamId)
{
    TestContext context;
    TestOperation operation;
    operation.SetExecuteStreamId(2);

    EXPECT_EQ(operation.GetExecuteStream(&context), nullptr);
}
