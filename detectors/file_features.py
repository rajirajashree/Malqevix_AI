"""The 27 input features of the CSV malware model, in the exact order used during training/scaling."""

FEATURE_COLUMNS = [
    'millisecond', 'state', 'usage_counter', 'prio', 'static_prio', 'normal_prio', 'policy', 'vm_pgoff', 'task_size',
    'cached_hole_size', 'free_area_cache', 'mm_users', 'map_count', 'hiwater_rss', 'total_vm', 'reserved_vm', 'nr_ptes',
    'end_data', 'last_interval', 'nivcsw', 'min_flt', 'fs_excl_counter', 'lock', 'stime', 'gtime', 'cgtime', 'signal_nvcsw',
]

# Columns present in the public dataset that are NOT model inputs. If a file contains them they are kept in the
# output CSV but never fed to the model (the original training notebook dropped the last six).
PASSTHROUGH_COLUMNS = ['hash', 'classification', 'vm_truncate_count', 'shared_vm', 'exec_vm', 'nvcsw', 'maj_flt', 'utime']
