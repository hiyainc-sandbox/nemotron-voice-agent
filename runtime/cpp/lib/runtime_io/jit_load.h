#pragma once

#include <torch/script.h>

#include <string>

torch::jit::Module load_jit_serialized(const std::string& path);
