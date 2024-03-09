#pragma once

#include <nlohmann/json.hpp>
using json = nlohmann::json;

struct Conf {
  json local_conf;
  int niters;
  bool validation;
  std::string output_fn;
};

extern Conf conf;