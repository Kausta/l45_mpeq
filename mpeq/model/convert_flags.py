# Convert Abseil flags to a json config file

from absl import flags
import json

# # paste list of flags
# flags.DEFINE_list('algorithms', ['matrix_chain_order'], 'Which algorithms to run.')
# ...

flag_values_dict = {}
for flag in flags.FLAGS:
    flag_values_dict[flag] = flags.FLAGS[flag].value

with open("clrs_train_config.json", "w") as f:
    json.dump(flag_values_dict, f)
