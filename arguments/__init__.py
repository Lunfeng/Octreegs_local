#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for orig_key, value in list(vars(self).items()):
            shorthand = False
            help_msg = None
            default_value = value

            if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], str):
                default_value, help_msg = value
                setattr(self, orig_key, default_value)

            if orig_key.startswith("_"):
                shorthand = True
                key = orig_key[1:]
            else:
                key = orig_key

            arg_default = default_value if not fill_none else None
            help_text = None
            if help_msg is not None:
                help_text = f"{help_msg} (default: {default_value})"

            t = type(default_value)
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=arg_default, action="store_true", help=help_text)
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=arg_default, type=t, help=help_text)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=arg_default, action="store_true", help=help_text)
                else:
                    group.add_argument("--" + key, default=arg_default, type=t, help=help_text)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.feat_dim = 32
        self.n_offsets = 10
        self.fork = 2

        self.use_feat_bank = False
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = 1
        self.white_background = False
        self.random_background = False
        self.resolution_scales = [1.0]

        self.data_device = "cuda"
        self.eval = False
        self.ds = 1
        self.ratio = 1  # sampling the input point cloud
        self.undistorted = False

        self.appearance_dim = 32
        self.add_opacity_dist = False
        self.add_cov_dist = False
        self.add_color_dist = False
        self.add_level = False

        self.extend = 1.1
        self.dist2level = 'round'
        self.base_layer = -1  # -1(adaptive) or 10 (default) or 0 ~
        self.visible_threshold = 0.0  # -1(adaptive) or 0.0 ~ 1.0
        self.update_ratio = 0.2

        self.progressive = False
        self.dist_ratio = 0.999  # 0.99/0.999
        self.levels = -1  # -1(adaptive) or 0 ~
        self.init_level = -1  # -1(adaptive) or 0 ~ levels-1
        self.extra_ratio = 0.25
        self.extra_up = 0.01

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 40_000
        self.position_lr_init = 0.0
        self.position_lr_final = 0.0
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = self.iterations

        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = self.iterations

        self.feature_lr = 0.0075
        self.opacity_lr = 0.02
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002

        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = self.iterations

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = self.iterations

        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = self.iterations

        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = self.iterations

        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = self.iterations

        self.appearance_lr_init = 0.05
        self.appearance_lr_final = 0.0005
        self.appearance_lr_delay_mult = 0.01
        self.appearance_lr_max_steps = self.iterations

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2

        # for anchor densification
        self.start_stat = 500
        self.update_from = 1500
        self.coarse_iter = 10000
        self.coarse_factor = 1.5
        self.update_interval = 100
        self.update_until = 25000
        self.update_anchor = True

        self.min_opacity = 0.005
        self.success_threshold = 0.8
        self.densify_grad_threshold = 0.0002

        self.spa_preset = ("minimal", "SPA configuration preset: off (disabled), minimal (pause densify), or full (keep densify)")
        self.spa_enable = (False, "Enable SPA-based sparsity management and proximal updates")
        self.spa_start_iter = (5000, "Iteration to start SPA regularization steps")
        self.spa_stop_iter = (8000, "Iteration to stop SPA regularization steps")
        self.spa_delta_start = (3e-4, "Initial SPA quadratic penalty strength")
        self.spa_delta_end = (1e-3, "Final SPA quadratic penalty strength")
        self.spa_interval_warm = (60, "Iteration interval between SPA steps during warm-up phase")
        self.spa_interval_stable = (40, "Iteration interval between SPA steps after warm-up")
        self.quota_update_interval = (1000, "Iterations between SPA quota recalculations")
        self.kappa_total = (300000, "Total target capacity for active SPA elements; set to 0 to defer to keep_ratio")
        self.keep_ratio = (0.0, "Optional keep ratio for SPA capacity when kappa_total is unset (set >0 to enable)")
        self.alpha_occupancy = (0.6, "Blend weight between visibility and gradient statistics for SPA quotas")
        self.anchor_m_min = (1, "Minimum slots reserved per active anchor during SPA projection")
        self.age_grace_iters = (2000, "Iterations granting young anchors guaranteed SPA slots")
        self.new_level_bootstrap_ratio = (0.1, "Fraction of total capacity reserved for newly activated levels")
        self.hot_anchor_boost = (1.0, "Score multiplier applied to hot anchors during SPA selection")
        self.hysteresis_M_out = (3, "Consecutive misses before marking SPA entries for pruning")
        self.hysteresis_M_in = (2, "Consecutive hits required to revive SPA entries from pruning")
        self.spa_keep_densify = (False, "Keep densification active during SPA and resize SPA buffers when counts grow")

        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)