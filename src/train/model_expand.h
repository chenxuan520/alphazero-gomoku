#pragma once

#include "cnn/policy_value_resnet.h"

namespace az {

// Embeds a smaller PolicyValueResNet into a wider/deeper compatible network.
// Existing channels are copied exactly, added channels are zero, and added
// residual blocks are initialized as identity maps. This preserves the source
// inference function while exposing new trainable capacity.
bool ExpandPolicyValueResNet(deeplearning::PolicyValueResNet &destination,
                             deeplearning::PolicyValueResNet &source);

} // namespace az
