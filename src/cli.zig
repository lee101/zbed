const std = @import("std");
const zbed = @import("zbed");
const gpu = zbed.gpu;
const EmbedModel = zbed.EmbedModel;
const EmbedScratch = zbed.EmbedScratch;
const Index = zbed.Index;
const QuantizedEmbedding = zbed.QuantizedEmbedding;
const SearchResult = zbed.SearchResult;
const WalkOptions = zbed.WalkOptions;

const MAX_RESULTS = 64;

const ParsedOptions = struct {
    target_path: []const u8 = ".",
    limit: usize = 10,
    threshold: f32 = 0.3,
    model_dir: ?[]const u8 = null,
    search_binaries: bool = false,
    use_gpu: bool = false,
};

pub fn runCli(program_name: []const u8) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        printUsage(program_name);
        return;
    }

    const command = args[1];
    if (std.mem.eql(u8, command, "index")) {
        try cmdIndex(allocator, program_name, args);
    } else if (std.mem.eql(u8, command, "status")) {
        try cmdStatus(allocator, args);
    } else if (std.mem.eql(u8, command, "bench")) {
        try cmdBench(allocator, args);
    } else if (std.mem.eql(u8, command, "embed")) {
        try cmdEmbed(allocator, program_name, args);
    } else if (std.mem.eql(u8, command, "serve")) {
        try cmdServe(allocator, args);
    } else if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "-h")) {
        printUsage(program_name);
    } else {
        try cmdSearch(allocator, program_name, args);
    }
}

fn stdout() std.fs.File.DeprecatedWriter {
    return std.fs.File.stdout().deprecatedWriter();
}

fn stderr() std.fs.File.DeprecatedWriter {
    return std.fs.File.stderr().deprecatedWriter();
}

fn printUsage(program_name: []const u8) void {
    stdout().print(
        \\{s} - semantic filesystem search with int8/512 static embeddings
        \\
        \\Usage:
        \\  {s} <query>              Search indexed files and file names
        \\  {s} index [path]         Build a quantized search index
        \\  {s} status [path]        Show index statistics
        \\  {s} embed <text>         Run one real embedding inference
        \\  {s} serve [--port N]      HTTP server with inotify live reindex
        \\  {s} bench                Run local embedding/search benchmarks
        \\  {s} help                 Show this help message
        \\
        \\Options:
        \\  -p, --path PATH          Search or index path (default: .)
        \\  -l, --limit N            Maximum results to show (default: 10)
        \\  -t, --threshold F        Similarity threshold (default: 0.3)
        \\  -m, --model-dir DIR      Path to model directory
        \\      --search-binaries    Index binary/media files by filename only
        \\      --gpu                Request GPU backend when available
        \\
        \\Server endpoints:
        \\  POST /search             {{"query":"text","k":10}}
        \\  POST /index              {{"documents":[{{"text":"...","id":1}}]}}
        \\  POST /rebuild            Full reindex from filesystem
        \\  POST /similarity         {{"text1":"a","text2":"b"}}
        \\  GET  /status             Index stats + watch status
        \\
    , .{ program_name, program_name, program_name, program_name, program_name, program_name, program_name, program_name }) catch {};
}

fn parseCommonOptions(args: []const []const u8, start_idx: usize) ParsedOptions {
    var opts = ParsedOptions{};
    var i = start_idx;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if ((std.mem.eql(u8, arg, "--limit") or std.mem.eql(u8, arg, "-l")) and i + 1 < args.len) {
            opts.limit = std.fmt.parseInt(usize, args[i + 1], 10) catch opts.limit;
            i += 1;
        } else if ((std.mem.eql(u8, arg, "--threshold") or std.mem.eql(u8, arg, "-t")) and i + 1 < args.len) {
            opts.threshold = std.fmt.parseFloat(f32, args[i + 1]) catch opts.threshold;
            i += 1;
        } else if ((std.mem.eql(u8, arg, "--model-dir") or std.mem.eql(u8, arg, "-m")) and i + 1 < args.len) {
            opts.model_dir = args[i + 1];
            i += 1;
        } else if ((std.mem.eql(u8, arg, "--path") or std.mem.eql(u8, arg, "-p")) and i + 1 < args.len) {
            opts.target_path = args[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, arg, "--search-binaries")) {
            opts.search_binaries = true;
        } else if (std.mem.eql(u8, arg, "--gpu")) {
            opts.use_gpu = true;
        }
    }
    return opts;
}

fn findModelDir(allocator: std.mem.Allocator) ![]const u8 {
    if (std.posix.getenv("ZBED_MODEL_PATH")) |path| return try allocator.dupe(u8, path);

    const candidates = [_][]const u8{
        "model",
        "../model",
        "../gobed/model",
        "../../model",
    };

    for (candidates) |candidate| {
        var buf: [4096]u8 = undefined;
        const check_path = std.fmt.bufPrint(&buf, "{s}/tokenizer.json", .{candidate}) catch continue;
        std.fs.cwd().access(check_path, .{}) catch continue;
        return try allocator.dupe(u8, candidate);
    }

    if (std.posix.getenv("HOME")) |home| {
        var buf: [4096]u8 = undefined;
        const home_model = std.fmt.bufPrint(&buf, "{s}/.zbed/model", .{home}) catch return error.ModelNotFound;
        var check_buf: [4096]u8 = undefined;
        const check_path = std.fmt.bufPrint(&check_buf, "{s}/tokenizer.json", .{home_model}) catch return error.ModelNotFound;
        std.fs.cwd().access(check_path, .{}) catch return error.ModelNotFound;
        return try allocator.dupe(u8, home_model);
    }

    return error.ModelNotFound;
}

fn tryLoadModel(allocator: std.mem.Allocator, opts: ParsedOptions) !EmbedModel {
    const model_dir = if (opts.model_dir) |dir| try allocator.dupe(u8, dir) else try findModelDir(allocator);
    defer allocator.free(model_dir);

    var model = EmbedModel.init(allocator);
    try model.loadFromDir(model_dir);
    return model;
}

fn loadModel(allocator: std.mem.Allocator, opts: ParsedOptions) !EmbedModel {
    return tryLoadModel(allocator, opts) catch {
        try stderr().print("Error: model not found. Run ./setup.sh or set ZBED_MODEL_PATH.\n", .{});
        std.process.exit(1);
    };
}

fn maybeReportGpuFallback(opts: ParsedOptions) !void {
    if (opts.use_gpu) {
        if (!gpu.cudaEnabled()) {
            try stderr().print("GPU backend requested, but this build was compiled without CUDA support. Rebuild with -Dcuda=true.\n", .{});
        } else if (!gpu.isAvailable()) {
            try stderr().print("GPU backend requested, but no CUDA device/backend is currently available. Using CPU path.\n", .{});
        }
    }
}

fn initGpuEmbedder(model: *const EmbedModel) ?gpu.GpuEmbedder {
    if (!gpu.isAvailable() or model.embed_dim != 512) return null;
    return gpu.GpuEmbedder.init(model) catch |err| {
        stderr().print("GPU embedder init failed: {s} ({})\n", .{ gpu.lastErrorMessage(), err }) catch {};
        return null;
    };
}

fn initGpuSearchIndex(allocator: std.mem.Allocator, idx: *const Index) ?gpu.GpuSearchIndex {
    if (!gpu.isAvailable() or idx.dim != 512) return null;
    return gpu.GpuSearchIndex.init(allocator, idx) catch |err| {
        stderr().print("GPU search init failed: {s} ({})\n", .{ gpu.lastErrorMessage(), err }) catch {};
        return null;
    };
}

fn cmdSearch(allocator: std.mem.Allocator, program_name: []const u8, args: []const []const u8) !void {
    _ = program_name;
    const opts = parseCommonOptions(args, 1);
    try maybeReportGpuFallback(opts);

    var query_parts: std.ArrayListUnmanaged([]const u8) = .{};
    defer query_parts.deinit(allocator);

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.startsWith(u8, arg, "-")) {
            if ((std.mem.eql(u8, arg, "--limit") or std.mem.eql(u8, arg, "-l") or
                std.mem.eql(u8, arg, "--threshold") or std.mem.eql(u8, arg, "-t") or
                std.mem.eql(u8, arg, "--model-dir") or std.mem.eql(u8, arg, "-m") or
                std.mem.eql(u8, arg, "--path") or std.mem.eql(u8, arg, "-p")) and i + 1 < args.len)
            {
                i += 1;
            }
            continue;
        }
        try query_parts.append(allocator, arg);
    }

    if (query_parts.items.len == 0) {
        try stderr().print("Error: no search query provided.\n", .{});
        return;
    }

    const query = try std.mem.join(allocator, " ", query_parts.items);
    defer allocator.free(query);

    var model = try loadModel(allocator, opts);
    defer model.deinit();

    var idx = Index.init(allocator, model.embed_dim);
    defer idx.deinit();
    var index_gpu_embedder = if (opts.use_gpu) initGpuEmbedder(&model) else null;
    defer if (index_gpu_embedder) |*gpu_embedder| gpu_embedder.deinit();

    if (Index.exists(opts.target_path)) {
        try idx.load(opts.target_path);
    } else {
        const walk_opts = WalkOptions{
            .search_binaries = opts.search_binaries,
            .gpu_embedder = if (index_gpu_embedder) |*gpu_embedder| gpu_embedder else null,
        };
        try stderr().print("No index found. Building one for {s}...\n", .{opts.target_path});
        try zbed.index.walkAndIndex(allocator, opts.target_path, &model, &idx, walk_opts, null);
        try idx.save(opts.target_path);
    }

    if (idx.count() == 0) {
        try stdout().print("No documents indexed.\n", .{});
        return;
    }

    var query_embedding = QuantizedEmbedding{};
    var maybe_gpu_embedder = if (opts.use_gpu) initGpuEmbedder(&model) else null;
    defer if (maybe_gpu_embedder) |*gpu_embedder| gpu_embedder.deinit();
    var query_scratch = EmbedScratch{};

    const valid_tokens = if (maybe_gpu_embedder) |*gpu_embedder|
        gpu_embedder.embedQuantized(&model, query, &query_scratch, &query_embedding) catch 0
    else
        model.embedQuantized(query, &query_embedding);
    if (valid_tokens == 0) {
        try stdout().print("Query produced no valid tokens.\n", .{});
        return;
    }

    const search_index = idx.buildSearchIndex();
    var maybe_gpu_search = if (opts.use_gpu) initGpuSearchIndex(allocator, &idx) else null;
    defer if (maybe_gpu_search) |*gpu_search| gpu_search.deinit();
    var results: [MAX_RESULTS]SearchResult = undefined;
    const n_results = if (maybe_gpu_search) |*gpu_search|
        gpu_search.search(&query_embedding, @min(opts.limit, MAX_RESULTS), opts.threshold, &results) catch search_index.search(
            query_embedding.data[0..model.embed_dim],
            query_embedding.norm,
            @min(opts.limit, MAX_RESULTS),
            opts.threshold,
            &results,
        )
    else
        search_index.search(
            query_embedding.data[0..model.embed_dim],
            query_embedding.norm,
            @min(opts.limit, MAX_RESULTS),
            opts.threshold,
            &results,
        );

    if (n_results == 0) {
        try stdout().print("No results found above threshold {d:.2}.\n", .{opts.threshold});
        return;
    }

    try stdout().print("{d} result(s) for \"{s}\":\n\n", .{ n_results, query });
    for (results[0..n_results], 0..) |result, idx_pos| {
        const doc = idx.documents.items[result.doc_idx];
        switch (doc.kind) {
            .path => try stdout().print("  {d}. [{d:.4}] {s}\n     filename: {s}\n\n", .{ idx_pos + 1, result.score, doc.file_path, doc.content }),
            .binary => try stdout().print("  {d}. [{d:.4}] {s}\n     binary filename: {s}\n\n", .{ idx_pos + 1, result.score, doc.file_path, doc.content }),
            .text => try stdout().print("  {d}. [{d:.4}] {s}:{d}\n     {s}\n\n", .{ idx_pos + 1, result.score, doc.file_path, doc.line_num, doc.content }),
        }
    }
}

fn cmdIndex(allocator: std.mem.Allocator, program_name: []const u8, args: []const []const u8) !void {
    _ = program_name;
    var opts = parseCommonOptions(args, 2);
    try maybeReportGpuFallback(opts);

    if (args.len > 2 and args[2].len > 0 and args[2][0] != '-') opts.target_path = args[2];

    var model = try loadModel(allocator, opts);
    defer model.deinit();

    try stderr().print("Indexing {s}...\n", .{opts.target_path});
    var timer = try std.time.Timer.start();

    var idx = Index.init(allocator, model.embed_dim);
    defer idx.deinit();

    var maybe_gpu_embedder = if (opts.use_gpu) initGpuEmbedder(&model) else null;
    defer if (maybe_gpu_embedder) |*gpu_embedder| gpu_embedder.deinit();

    const walk_opts = WalkOptions{
        .search_binaries = opts.search_binaries,
        .gpu_embedder = if (maybe_gpu_embedder) |*gpu_embedder| gpu_embedder else null,
    };
    const progress = struct {
        fn callback(count: usize) void {
            if (count > 0 and count % 1000 == 0) {
                std.fs.File.stderr().deprecatedWriter().print("\r  {d} docs indexed...", .{count}) catch {};
            }
        }
    }.callback;

    try zbed.index.walkAndIndex(allocator, opts.target_path, &model, &idx, walk_opts, &progress);
    try idx.save(opts.target_path);

    const elapsed_ms = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
    try stderr().print("\r", .{});
    const summary = try idx.summarize(allocator);
    try stdout().print("Indexed {d} docs from {d} files in {d:.1} ms\n", .{ idx.count(), summary.files, elapsed_ms });
    try stdout().print("  path docs: {d}, content lines: {d}, binary filename docs: {d}\n", .{ summary.path_docs, summary.text_docs, summary.binary_docs });
    try stdout().print("  saved: {s}/.zbed/index.bin\n", .{opts.target_path});
}

fn cmdStatus(allocator: std.mem.Allocator, args: []const []const u8) !void {
    var path: []const u8 = ".";
    if (args.len > 2 and args[2].len > 0 and args[2][0] != '-') path = args[2];

    if (!Index.exists(path)) {
        try stdout().print("No index found at {s}/.zbed/index.bin\n", .{path});
        return;
    }

    var idx = Index.init(allocator, 1);
    defer idx.deinit();
    try idx.load(path);

    const summary = try idx.summarize(allocator);
    const quant_bytes = idx.count() * idx.dim + idx.count() * @sizeOf(f32) * 2;
    const quant_mb = @as(f64, @floatFromInt(quant_bytes)) / (1024.0 * 1024.0);

    try stdout().print("Index: {s}/.zbed/index.bin\n", .{path});
    try stdout().print("  documents: {d}\n", .{idx.count()});
    try stdout().print("  files: {d}\n", .{summary.files});
    try stdout().print("  dimensions: {d}\n", .{idx.dim});
    try stdout().print("  path docs: {d}\n", .{summary.path_docs});
    try stdout().print("  content docs: {d}\n", .{summary.text_docs});
    try stdout().print("  binary docs: {d}\n", .{summary.binary_docs});
    try stdout().print("  quantized embedding store: {d:.2} MB\n", .{quant_mb});
}

fn cmdEmbed(allocator: std.mem.Allocator, program_name: []const u8, args: []const []const u8) !void {
    const opts = parseCommonOptions(args, 2);
    try maybeReportGpuFallback(opts);

    if (args.len < 3) {
        printUsage(program_name);
        return;
    }

    const text = args[2];
    var model = try loadModel(allocator, opts);
    defer model.deinit();

    var quantized = QuantizedEmbedding{};
    var maybe_gpu_embedder = if (opts.use_gpu) initGpuEmbedder(&model) else null;
    defer if (maybe_gpu_embedder) |*gpu_embedder| gpu_embedder.deinit();

    var scratch = EmbedScratch{};
    const valid = if (maybe_gpu_embedder) |*gpu_embedder|
        gpu_embedder.embedQuantized(&model, text, &scratch, &quantized) catch model.embedQuantized(text, &quantized)
    else
        model.embedQuantized(text, &quantized);
    try stdout().print("tokens={d} dim={d} scale={d:.6} norm={d:.3}\n", .{ valid, model.embed_dim, quantized.scale, quantized.norm });
    try stdout().print("first16=", .{});
    const show = @min(model.embed_dim, 16);
    for (quantized.data[0..show], 0..) |value, i| {
        if (i != 0) try stdout().print(",", .{});
        try stdout().print("{d}", .{value});
    }
    try stdout().print("\n", .{});
}

fn cmdBench(allocator: std.mem.Allocator, args: []const []const u8) !void {
    const opts = parseCommonOptions(args, 2);

    var model = tryLoadModel(allocator, opts) catch {
        try stderr().print("Model not available. Run ./setup.sh for full inference benchmarks.\n", .{});
        return;
    };
    defer model.deinit();

    const texts = [_][]const u8{
        "error handling in production",
        "database connection pooling",
        "user authentication middleware",
        "REST API endpoint design",
        "media file search mp3 opus mp4",
        "binary filename indexing rules",
    };

    var timer = try std.time.Timer.start();
    var quantized = QuantizedEmbedding{};
    const iters: usize = 200;
    var cpu_embed_checksum: i64 = 0;
    for (0..iters) |_| {
        for (texts) |text| {
            const valid = model.embedQuantized(text, &quantized);
            cpu_embed_checksum += @intCast(valid);
            cpu_embed_checksum += quantized.data[0];
            cpu_embed_checksum += quantized.data[1];
        }
    }

    const elapsed_ns = timer.read();
    const total = iters * texts.len;
    const us_per = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(total)) / 1000.0;
    try stdout().print("CPU embedding benchmark: {d} runs, {d:.1} us/embed (checksum={d})\n", .{ total, us_per, cpu_embed_checksum });

    if (opts.use_gpu and gpu.isAvailable() and model.embed_dim == 512) {
        if (gpu.deviceName(allocator, 0) catch null) |name| {
            defer allocator.free(name);
            try stdout().print("CUDA device: {s}\n", .{name});
        }

        var gpu_embedder = initGpuEmbedder(&model) orelse return;
        defer gpu_embedder.deinit();
        var scratch = EmbedScratch{};
        var gpu_embed_checksum: i64 = 0;

        timer.reset();
        for (0..iters) |_| {
            for (texts) |text| {
                const valid = gpu_embedder.embedQuantized(&model, text, &scratch, &quantized) catch 0;
                gpu_embed_checksum += @intCast(valid);
                gpu_embed_checksum += quantized.data[0];
                gpu_embed_checksum += quantized.data[1];
            }
        }
        const gpu_embed_us = @as(f64, @floatFromInt(timer.read())) / @as(f64, @floatFromInt(total)) / 1000.0;
        try stdout().print("GPU embedding benchmark: {d} runs, {d:.1} us/embed (checksum={d})\n", .{ total, gpu_embed_us, gpu_embed_checksum });
    }

    var idx = Index.init(allocator, model.embed_dim);
    defer idx.deinit();
    for (texts, 0..) |text, i| {
        _ = model.embedQuantized(text, &quantized);
        try idx.addDocumentQuantized("bench.txt", @intCast(i + 1), text, .text, quantized.data[0..model.embed_dim], quantized.scale, quantized.norm);
    }

    const search_index = idx.buildSearchIndex();
    var results: [8]SearchResult = undefined;
    var cpu_search_checksum: f64 = 0;
    timer.reset();
    for (0..1000) |_| {
        _ = model.embedQuantized("binary filename audio search", &quantized);
        const found = search_index.search(quantized.data[0..model.embed_dim], quantized.norm, 3, 0.0, &results);
        cpu_search_checksum += @floatFromInt(found);
        if (found > 0) cpu_search_checksum += results[0].score;
    }
    const search_us = @as(f64, @floatFromInt(timer.read())) / 1000.0 / 1000.0;
    try stdout().print("CPU search benchmark: 1000 runs, {d:.1} us/search (checksum={d:.3})\n", .{ search_us, cpu_search_checksum });

    if (opts.use_gpu and gpu.isAvailable() and idx.dim == 512) {
        var gpu_search = initGpuSearchIndex(allocator, &idx) orelse return;
        defer gpu_search.deinit();
        var gpu_embedder = initGpuEmbedder(&model) orelse return;
        defer gpu_embedder.deinit();
        var scratch = EmbedScratch{};
        var gpu_search_checksum: f64 = 0;

        timer.reset();
        for (0..1000) |_| {
            _ = gpu_embedder.embedQuantized(&model, "binary filename audio search", &scratch, &quantized) catch 0;
            const found = gpu_search.search(&quantized, 3, 0.0, &results) catch 0;
            gpu_search_checksum += @floatFromInt(found);
            if (found > 0) gpu_search_checksum += results[0].score;
        }
        const gpu_search_us = @as(f64, @floatFromInt(timer.read())) / 1000.0 / 1000.0;
        try stdout().print("GPU search benchmark: 1000 runs, {d:.1} us/search (checksum={d:.3})\n", .{ gpu_search_us, gpu_search_checksum });
    }
}

fn cmdServe(allocator: std.mem.Allocator, args: []const []const u8) !void {
    const opts = parseCommonOptions(args, 2);
    try maybeReportGpuFallback(opts);

    var port: u16 = 8080;
    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        if ((std.mem.eql(u8, args[i], "--port") or std.mem.eql(u8, args[i], "-P")) and i + 1 < args.len) {
            port = std.fmt.parseInt(u16, args[i + 1], 10) catch 8080;
            i += 1;
        }
    }

    var model = try loadModel(allocator, opts);
    defer model.deinit();

    var idx = Index.init(allocator, model.embed_dim);
    defer idx.deinit();

    if (Index.exists(opts.target_path)) {
        try stderr().print("Loading index from {s}/.zbed/index.bin\n", .{opts.target_path});
        try idx.load(opts.target_path);
        try stderr().print("{d} docs loaded\n", .{idx.count()});
    } else {
        try stderr().print("Building index for {s}...\n", .{opts.target_path});
        const walk_opts = WalkOptions{ .search_binaries = opts.search_binaries };
        try zbed.index.walkAndIndex(allocator, opts.target_path, &model, &idx, walk_opts, null);
        try idx.save(opts.target_path);
        try stderr().print("{d} docs indexed\n", .{idx.count()});
    }

    const walk_opts = WalkOptions{ .search_binaries = opts.search_binaries };
    var state = zbed.ServerState.init(allocator, &model, &idx, opts.target_path, walk_opts);
    defer state.deinit();

    try zbed.server.run(allocator, &state, port);
}
