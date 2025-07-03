def get_prefix_str(args):
    use_data_aug = args.use_data_aug
    sep_query_head = args.sep_query_head
    kb_size = args.kb_size
    dynamic_kb_size = args.dynamic_kb_size

    if dynamic_kb_size is not None:
        kb_size = "dynamic"  # Random size

    duplicate_true_kb = args.duplicate_true_kb
    length_invariance = args.length_invariance
    outlier_ratio = args.outlier_num
    use_outlier = outlier_ratio != -1
    multi_entities = args.multi_entities
    use_extended_qa = args.use_extended_qa
    kb_token_layer_frequency = args.kb_token_layer_frequency
    lr = args.lr

    prefix_string = f"stage1_lr_{str(lr).replace('.', 'p')}"
    if kb_token_layer_frequency is not None:
        prefix_string += f"KBTokenLayerFreq{kb_token_layer_frequency}"
    if use_extended_qa:
        prefix_string += "UseExtendedQA"
    if multi_entities is not None:
        prefix_string += f"MultiEntities{multi_entities}"
    if use_outlier:
        prefix_string += f"UseOutlier{outlier_ratio}"
    if length_invariance:
        prefix_string += "LengthInvariant"
    if not duplicate_true_kb:
        prefix_string += "NoDuplicate"
    if kb_size is not None:
        prefix_string += f"KBSize{kb_size}"
    if sep_query_head:
        prefix_string += "SepQueryHead"
    if use_data_aug:
        prefix_string += "UseDataAug"
    return prefix_string


def get_step_config(
    current_accum_step: int,
    total_accum_step: int,
    use_data_aug: bool,
    outlier_num: int,
    multi_entities: int | None,
    use_extended_qa: bool,
):
    """
    Our instruction tuning dataset is composed of different types of instructions.
    Strategies:
    Outlier QA takes the last `outlier_num` accum steps;
    Multiple entites QA (if included) takes 1/3 of the rest accum_steps;
    Extended QA (if included) takes 1/3 of the rest accum_steps;
    Standard QA takes the rest.
    """
    config = {}
    config["use_data_aug"] = use_data_aug
    config["include_outlier"] = False
    config["multi_entities"] = None
    config["use_extended_qa"] = False
    include_outlier = current_accum_step >= total_accum_step - 1 - outlier_num
    # Decide to include outlier and has reached the time
    if include_outlier:
        config["include_outlier"] = True
        return config
    if current_accum_step % 3 == 0:
        # multi_entities could be None,
        # in which case we just use standard QA
        config["multi_entities"] = multi_entities
        return config
    if current_accum_step % 3 == 1:
        config["use_extended_qa"] = use_extended_qa
        return config
    return config
