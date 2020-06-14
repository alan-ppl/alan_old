Sampling always happens with a single extra dimension, even if we e.g. eventually sum out discrete variables, so this should be simple.
Then, we compute log_prob on a second pass.
  log_prob dynamically rearranges samples before returning them.
  fn should be an object, with state

