#include "train/model_expand.h"

#include <algorithm>

namespace az {
namespace {

using deeplearning::BatchNorm2D;
using deeplearning::BatchedConv2D;
using deeplearning::FloatLinear;

bool ExpandConv(BatchedConv2D &destination, const BatchedConv2D &source) {
  const auto &dst = destination.config();
  const auto &src = source.config();
  if (dst.input_channels_ < src.input_channels_ ||
      dst.output_channels_ < src.output_channels_ ||
      dst.kernel_height_ != src.kernel_height_ ||
      dst.kernel_width_ != src.kernel_width_ || dst.stride_ != src.stride_ ||
      dst.padding_ != src.padding_ || dst.use_bias_ != src.use_bias_) {
    return false;
  }
  // Preserve random weights for newly added output channels. For an existing
  // output channel, connections from added inputs must start at zero so the
  // source function is unchanged.
  std::vector<float> weight = destination.weight();
  for (int output = 0; output < src.output_channels_; ++output) {
    for (int input = src.input_channels_; input < dst.input_channels_; ++input) {
      for (int row = 0; row < dst.kernel_height_; ++row) {
        for (int column = 0; column < dst.kernel_width_; ++column) {
          const std::size_t index =
              ((static_cast<std::size_t>(output) * dst.input_channels_ + input) *
                   dst.kernel_height_ +
               row) *
                  dst.kernel_width_ +
              column;
          weight[index] = 0.0f;
        }
      }
    }
  }
  for (int output = 0; output < src.output_channels_; ++output) {
    for (int input = 0; input < src.input_channels_; ++input) {
      for (int row = 0; row < src.kernel_height_; ++row) {
        for (int column = 0; column < src.kernel_width_; ++column) {
          const std::size_t src_index =
              ((static_cast<std::size_t>(output) * src.input_channels_ + input) *
                   src.kernel_height_ +
               row) *
                  src.kernel_width_ +
              column;
          const std::size_t dst_index =
              ((static_cast<std::size_t>(output) * dst.input_channels_ + input) *
                   dst.kernel_height_ +
               row) *
                  dst.kernel_width_ +
              column;
          weight[dst_index] = source.weight()[src_index];
        }
      }
    }
  }
  std::vector<float> bias = destination.bias();
  std::copy(source.bias().begin(), source.bias().end(), bias.begin());
  return destination.set_weight(weight) == BatchedConv2D::SUCCESS &&
         destination.set_bias(bias) == BatchedConv2D::SUCCESS;
}

bool ResetConv(BatchedConv2D &convolution) {
  return convolution.set_weight(
             std::vector<float>(convolution.weight().size(), 0.0f)) ==
             BatchedConv2D::SUCCESS &&
         convolution.set_bias(
             std::vector<float>(convolution.bias().size(), 0.0f)) ==
             BatchedConv2D::SUCCESS;
}

bool ExpandNorm(BatchNorm2D &destination, const BatchNorm2D &source) {
  const int src_channels = source.config().channels_;
  const int dst_channels = destination.config().channels_;
  if (dst_channels < src_channels) return false;
  std::vector<float> scale(dst_channels, 1.0f);
  std::vector<float> bias(dst_channels, 0.0f);
  std::vector<float> mean(dst_channels, 0.0f);
  std::vector<float> variance(dst_channels, 1.0f);
  std::copy(source.scale().begin(), source.scale().end(), scale.begin());
  std::copy(source.bias().begin(), source.bias().end(), bias.begin());
  std::copy(source.running_mean().begin(), source.running_mean().end(),
            mean.begin());
  std::copy(source.running_variance().begin(),
            source.running_variance().end(), variance.begin());
  return destination.set_scale(scale) == BatchNorm2D::SUCCESS &&
         destination.set_bias(bias) == BatchNorm2D::SUCCESS &&
         destination.set_running_mean(mean) == BatchNorm2D::SUCCESS &&
         destination.set_running_variance(variance) == BatchNorm2D::SUCCESS;
}

bool ResetNorm(BatchNorm2D &normalization) {
  const int channels = normalization.config().channels_;
  std::vector<float> bias(channels, 0.0f);
  return normalization.set_scale(std::vector<float>(channels, 1.0f)) ==
             BatchNorm2D::SUCCESS &&
         normalization.set_bias(bias) ==
             BatchNorm2D::SUCCESS &&
         normalization.set_running_mean(
             std::vector<float>(channels, 0.0f)) == BatchNorm2D::SUCCESS &&
         normalization.set_running_variance(
             std::vector<float>(channels, 1.0f)) == BatchNorm2D::SUCCESS;
}

bool CopyLinear(FloatLinear &destination, const FloatLinear &source) {
  if (destination.config().input_dim_ != source.config().input_dim_ ||
      destination.config().output_dim_ != source.config().output_dim_ ||
      destination.config().use_bias_ != source.config().use_bias_) {
    return false;
  }
  return destination.set_weight(source.weight()) == FloatLinear::SUCCESS &&
         destination.set_bias(source.bias()) == FloatLinear::SUCCESS;
}

bool ResetBlock(deeplearning::ResidualBlock2D &block) {
  // Keep conv1's independent random features, but zero conv2 so the block is
  // initially an exact identity. The first step learns conv2; then gradients
  // can flow into conv1 on subsequent steps.
  return ResetNorm(block.norm1()) && ResetConv(block.conv2()) &&
         ResetNorm(block.norm2());
}

} // namespace

bool ExpandPolicyValueResNet(deeplearning::PolicyValueResNet &destination,
                             deeplearning::PolicyValueResNet &source) {
  const auto &dst = destination.config();
  const auto &src = source.config();
  if (dst.input_channels_ != src.input_channels_ ||
      dst.board_height_ != src.board_height_ ||
      dst.board_width_ != src.board_width_ ||
      dst.policy_channels_ != src.policy_channels_ ||
      dst.policy_size_ != src.policy_size_ ||
      dst.value_channels_ != src.value_channels_ ||
      dst.value_hidden_dim_ != src.value_hidden_dim_ ||
      dst.trunk_channels_ < src.trunk_channels_ ||
      dst.residual_block_num_ < src.residual_block_num_ ||
      dst.batch_norm_epsilon_ != src.batch_norm_epsilon_ ||
      dst.batch_norm_momentum_ != src.batch_norm_momentum_) {
    return false;
  }
  if (!ExpandConv(destination.stem_conv(), source.stem_conv()) ||
      !ExpandNorm(destination.stem_norm(), source.stem_norm())) {
    return false;
  }
  for (int index = 0; index < src.residual_block_num_; ++index) {
    auto &dst_block = destination.blocks()[index];
    auto &src_block = source.blocks()[index];
    if (!ExpandConv(dst_block.conv1(), src_block.conv1()) ||
        !ExpandNorm(dst_block.norm1(), src_block.norm1()) ||
        !ExpandConv(dst_block.conv2(), src_block.conv2()) ||
        !ExpandNorm(dst_block.norm2(), src_block.norm2())) {
      return false;
    }
  }
  for (int index = src.residual_block_num_;
       index < dst.residual_block_num_; ++index) {
    if (!ResetBlock(destination.blocks()[index])) return false;
  }
  return ExpandConv(destination.policy_conv(), source.policy_conv()) &&
         ExpandNorm(destination.policy_norm(), source.policy_norm()) &&
         CopyLinear(destination.policy_linear(), source.policy_linear()) &&
         ExpandConv(destination.value_conv(), source.value_conv()) &&
         ExpandNorm(destination.value_norm(), source.value_norm()) &&
         CopyLinear(destination.value_hidden(), source.value_hidden()) &&
         CopyLinear(destination.value_output(), source.value_output());
}

} // namespace az
