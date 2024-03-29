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


# Convert Abseil flags to a (copy-able) dictionary

def print_dict_lines(my_dict):
    lines = []
    for key, value in my_dict.items():
        if isinstance(value, str):
            lines.append(f"'{key}': '{value}',")
        else:
            lines.append(f"'{key}': {value},")
    print("my_dict = {")
    for line in lines:
        print(f"\t{line}")
    print("}")
    
    
print_dict_lines(flag_values_dict)
