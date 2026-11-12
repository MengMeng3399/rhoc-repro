import json
import platform
import random
import statistics
import time


def benchmark(args):
    import torch

    from .cli import catalog_counts, graph_counts, queries_for, test_fingerprint
    from .data import load_catalog, split_invocations
    from .experiment import build_model, select_device, synchronize, write_report
    from .model import build_views
    from .text import BGEEncoder, DemoEncoder

    report = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("method") != "rhoc"
        or report.get("context") != "mixed"
        or report.get("variant") != "full"
    ):
        raise ValueError("The benchmark requires a full RHOC run in the mixed context.")

    seeds = [run["seed"] for run in report["runs"]]
    if not seeds:
        raise ValueError("The run report has no saved seeds.")
    seed = args.seed if args.seed is not None else seeds[0]
    if seed not in seeds:
        raise ValueError(f"Seed {seed} does not appear in the run report.")

    config = report["config"]
    synthetic = report["synthetic"]
    if synthetic and args.bge_checkpoint is not None:
        raise ValueError("A synthetic run requires the demo text encoder.")
    catalog = load_catalog(args.data)
    split_seed = config["split_seed"]
    splits = split_invocations(catalog, split_seed)
    queries = queries_for(splits, config, "test", (1, 2, 3), split_seed)
    if not queries or test_fingerprint(catalog, queries) != report["test_fingerprint"]:
        raise ValueError("The local data do not reproduce the run's test queries and catalog.")
    views = build_views(catalog, splits["train"], config["views"])
    if catalog_counts(catalog) != report["catalog"] or graph_counts(views) != report["graphs"]:
        raise ValueError("The local data do not reproduce the run's catalog and graph counts.")

    device = select_device(args.device)
    state = torch.load(args.run / f"seed-{seed}.pt", map_location="cpu", weights_only=True)
    model = build_model(views, config, report["variant"]).to(device=device, dtype=torch.float32).eval()
    model.load_state_dict(state)
    checkpoint = args.bge_checkpoint or report["text_checkpoint"]
    if synthetic:
        encoder = DemoEncoder(config["text_dim"])
    else:
        encoder = BGEEncoder(checkpoint, device, config["text_dim"])
        encoder.model.float()

    rng = random.Random(split_seed)
    measured = rng.sample(queries, min(args.queries, len(queries)))
    warmup = rng.sample(queries, min(args.warmup, len(queries)))

    def inputs(query):
        context = torch.tensor([query.context], device=device, dtype=torch.long)
        return catalog.requirements[query.app], context

    measured_inputs = [inputs(query) for query in measured]
    warmup_inputs = [inputs(query) for query in warmup]
    matmul_precision = torch.get_float32_matmul_precision()
    convolution_tf32 = None
    if device.type == "cuda":
        convolution_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.inference_mode():
            api_embeddings = model.api_embeddings()

            def select_top20(requirement, context):
                text = encoder.encode([requirement], 1, output_device=device)
                query = model.query_embeddings(api_embeddings, context, text)
                scores = query @ api_embeddings.T
                scores.scatter_(1, context, -torch.inf)
                return scores.topk(min(20, scores.size(1) - context.size(1)), dim=1)

            for requirement, context in warmup_inputs:
                select_top20(requirement, context)

            durations = []
            for requirement, context in measured_inputs:
                synchronize(device)
                start = time.perf_counter()
                select_top20(requirement, context)
                synchronize(device)
                durations.append(time.perf_counter() - start)
    finally:
        torch.set_float32_matmul_precision(matmul_precision)
        if convolution_tf32 is not None:
            torch.backends.cudnn.allow_tf32 = convolution_tf32

    write_report({
        "dataset": report["dataset"],
        "synthetic": synthetic,
        "context": report["context"],
        "seed": seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
        "dtype": "float32",
        "batch_size": 1,
        "text_checkpoint": checkpoint,
        "test_queries": len(queries),
        "warmup_queries": len(warmup_inputs),
        "measured_queries": len(measured_inputs),
        "mean_inference_ms_per_query": statistics.mean(durations) * 1_000,
    }, None)
