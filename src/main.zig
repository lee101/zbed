const std = @import("std");
const zbed = @import("zbed");

const EmbedModel = zbed.EmbedModel;
const Index = zbed.Index;
const SearchResult = zbed.SearchResult;

const MAX_RESULTS = 20;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        printUsage();
        return;
    }

    const command = args[1];

    if (std.mem.eql(u8, command, "index")) {
        try cmdIndex(allocator, args);
    } else if (std.mem.eql(u8, command, "bench")) {
        try cmdBench(allocator);
    } else if (std.mem.eql(u8, command, "status")) {
        try cmdStatus(allocator, args);
    } else if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "-h")) {
        printUsage();
    } else {
        // Default: treat as search query
        try cmdSearch(allocator, args);
    }
}

fn printUsage() void {
    const stdout = std.io.getStdOut().writer();
    stdout.print(
        \\zbed - fast semantic code search
        \\
        \\Usage:
        \\  zbed <query>            Search indexed files for semantically similar lines
        \\  zbed index [path]       Build semantic index for a directory (default: .)
        \\  zbed bench              Run embedding benchmark
        \\  zbed status [path]      Show index statistics
        \\  zbed help               Show this help message
        \\
        \\Options:
        \\  -l, --limit N           Maximum results to show (default: 10)
        \\  -t, --threshold F       Similarity threshold 0.0-1.0 (default: 0.3)
        \\  -m, --model-dir DIR     Path to model directory (default: model/)
        \\
        \\Examples:
        \\  zbed "error handling"   Search for code related to error handling
        \\  zbed index .            Build index for current directory
        \\  zbed index src/         Build index for src/ directory
        \\  zbed status             Show index stats for current directory
        \\  zbed bench              Run embedding performance benchmark
        \\
    , .{}) catch {};
}

fn findModelDir(allocator: std.mem.Allocator) ![]const u8 {

    // Check command-line --model-dir (would need to parse args more fully)
    // Check environment variable
    if (std.posix.getenv("ZBED_MODEL_PATH")) |p| return p;

    // Check common locations
    const candidates = [_][]const u8{
        "model",
        "../model",
        "../../model",
    };

    for (candidates) |candidate| {
        var buf: [4096]u8 = undefined;
        const check_path = std.fmt.bufPrint(&buf, "{s}/tokenizer.json", .{candidate}) catch continue;
        std.fs.cwd().access(check_path, .{}) catch continue;
        return try allocator.dupe(u8, candidate);
    }

    // Try home directory
    if (std.posix.getenv("HOME")) |home| {
        var buf: [4096]u8 = undefined;
        const home_model = std.fmt.bufPrint(&buf, "{s}/.zbed/model", .{home}) catch return error.ModelNotFound;
        var buf2: [4096]u8 = undefined;
        const check_path = std.fmt.bufPrint(&buf2, "{s}/tokenizer.json", .{home_model}) catch return error.ModelNotFound;
        std.fs.cwd().access(check_path, .{}) catch return error.ModelNotFound;
        return try allocator.dupe(u8, home_model);
    }

    return error.ModelNotFound;
}

fn tryLoadModel(allocator: std.mem.Allocator, args: []const []const u8) !EmbedModel {
    // Parse --model-dir from args
    var model_dir: ?[]const u8 = null;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if ((std.mem.eql(u8, args[i], "--model-dir") or std.mem.eql(u8, args[i], "-m")) and i + 1 < args.len) {
            model_dir = args[i + 1];
            i += 1;
        }
    }

    const dir = model_dir orelse try findModelDir(allocator);
    var model = EmbedModel.init(allocator);
    try model.loadFromDir(dir);
    return model;
}

fn loadModel(allocator: std.mem.Allocator, args: []const []const u8) !EmbedModel {
    return tryLoadModel(allocator, args) catch {
        const stderr = std.io.getStdErr().writer();
        try stderr.print("Error: Model not found. Run setup.sh to download the model.\n", .{});
        try stderr.print("Or set ZBED_MODEL_PATH environment variable.\n", .{});
        std.process.exit(1);
    };
}

fn cmdSearch(allocator: std.mem.Allocator, args: []const []const u8) !void {
    const stdout = std.io.getStdOut().writer();
    const stderr = std.io.getStdErr().writer();

    // Parse arguments
    var query_parts = std.ArrayList([]const u8).init(allocator);
    defer query_parts.deinit();
    var limit: usize = 10;
    var threshold: f32 = 0.3;
    var search_path: []const u8 = ".";

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if ((std.mem.eql(u8, args[i], "--limit") or std.mem.eql(u8, args[i], "-l")) and i + 1 < args.len) {
            limit = std.fmt.parseInt(usize, args[i + 1], 10) catch 10;
            i += 1;
        } else if ((std.mem.eql(u8, args[i], "--threshold") or std.mem.eql(u8, args[i], "-t")) and i + 1 < args.len) {
            threshold = std.fmt.parseFloat(f32, args[i + 1]) catch 0.3;
            i += 1;
        } else if ((std.mem.eql(u8, args[i], "--model-dir") or std.mem.eql(u8, args[i], "-m")) and i + 1 < args.len) {
            i += 1; // skip, handled by loadModel
        } else if (std.mem.eql(u8, args[i], "--path") or std.mem.eql(u8, args[i], "-p")) {
            if (i + 1 < args.len) {
                search_path = args[i + 1];
                i += 1;
            }
        } else {
            try query_parts.append(args[i]);
        }
    }

    if (query_parts.items.len == 0) {
        try stderr.print("Error: No search query provided.\n", .{});
        printUsage();
        return;
    }

    // Join query parts
    const query = try std.mem.join(allocator, " ", query_parts.items);
    defer allocator.free(query);

    // Load model
    var model = try loadModel(allocator, args);
    defer model.deinit();

    // Load or build index
    var idx = Index.init(allocator, model.embed_dim);
    defer idx.deinit();

    if (Index.exists(search_path)) {
        try stderr.print("Loading index from {s}/.zbed/index.bin...\n", .{search_path});
        try idx.load(search_path);
        try stderr.print("Loaded {d} documents.\n", .{idx.count()});
    } else {
        try stderr.print("No index found. Building index for {s}...\n", .{search_path});
        try zbed.index.walkAndIndex(allocator, search_path, &model, &idx, null);
        try stderr.print("Indexed {d} documents.\n", .{idx.count()});
        idx.save(search_path) catch |err| {
            try stderr.print("Warning: Could not save index: {}\n", .{err});
        };
    }

    if (idx.count() == 0) {
        try stderr.print("No documents indexed. Nothing to search.\n", .{});
        return;
    }

    // Build search index
    var flat = try idx.buildSearchIndex(allocator);
    defer flat.deinit();

    // Embed query
    var query_emb: [zbed.MAX_EMBED_DIM]f32 = undefined;
    const valid = model.embed(query, query_emb[0..model.embed_dim]);
    if (valid == 0) {
        try stderr.print("Warning: Query produced no valid tokens.\n", .{});
        return;
    }

    // Search
    var results: [MAX_RESULTS]SearchResult = undefined;
    const n_results = flat.search(query_emb[0..model.embed_dim], @min(limit, MAX_RESULTS), threshold, &results);

    if (n_results == 0) {
        try stdout.print("No results found above threshold {d:.2}.\n", .{threshold});
        return;
    }

    try stdout.print("\n{d} results for \"{s}\":\n\n", .{ n_results, query });

    for (results[0..n_results], 0..) |r, ri| {
        const doc = idx.documents.items[r.doc_idx];
        try stdout.print("  {d}. [{d:.4}] {s}:{d}\n", .{ ri + 1, r.score, doc.file_path, doc.line_num });
        try stdout.print("     {s}\n\n", .{doc.content});
    }
}

fn cmdIndex(allocator: std.mem.Allocator, args: []const []const u8) !void {
    const stdout = std.io.getStdOut().writer();
    const stderr = std.io.getStdErr().writer();

    var path: []const u8 = ".";
    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        if (args[i][0] != '-') {
            path = args[i];
            break;
        }
        if ((std.mem.eql(u8, args[i], "--model-dir") or std.mem.eql(u8, args[i], "-m")) and i + 1 < args.len) {
            i += 1;
        }
    }

    var model = try loadModel(allocator, args);
    defer model.deinit();

    try stderr.print("Indexing {s}...\n", .{path});

    var timer = try std.time.Timer.start();

    var idx = Index.init(allocator, model.embed_dim);
    defer idx.deinit();

    const progress = struct {
        fn callback(count: usize) void {
            if (count % 1000 == 0 and count > 0) {
                std.io.getStdErr().writer().print("\r  {d} lines indexed...", .{count}) catch {};
            }
        }
    }.callback;

    try zbed.index.walkAndIndex(allocator, path, &model, &idx, &progress);

    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;

    try stderr.print("\r", .{});
    try stdout.print("Indexed {d} lines in {d:.1} ms\n", .{ idx.count(), elapsed_ms });

    // Save index
    try idx.save(path);
    try stdout.print("Index saved to {s}/.zbed/index.bin\n", .{path});
}

fn cmdBench(allocator: std.mem.Allocator) !void {
    const stdout = std.io.getStdOut().writer();
    const stderr = std.io.getStdErr().writer();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var model = tryLoadModel(allocator, args) catch {
        try stderr.print("Model not available for benchmark. Running synthetic benchmark.\n", .{});
        try stderr.print("Run setup.sh to enable full embedding benchmark.\n\n", .{});
        try benchSynthetic(allocator);
        return;
    };
    defer model.deinit();

    const texts = [_][]const u8{
        "error handling in production",
        "database connection pooling",
        "user authentication middleware",
        "REST API endpoint design",
        "memory management optimization",
        "concurrent data structure implementation",
        "logging and monitoring setup",
        "configuration file parsing",
        "unit test coverage improvement",
        "deployment pipeline automation",
    };

    try stdout.print("Embedding benchmark ({d} texts, dim={d}):\n", .{ texts.len, model.embed_dim });

    var emb_buf: [zbed.MAX_EMBED_DIM]f32 = undefined;
    const n_iters: usize = 100;

    var timer = try std.time.Timer.start();

    for (0..n_iters) |_| {
        for (texts) |text| {
            _ = model.embed(text, emb_buf[0..model.embed_dim]);
        }
    }

    const elapsed_ns = timer.read();
    const total_embeds = n_iters * texts.len;
    const ns_per_embed = elapsed_ns / total_embeds;
    const us_per_embed = @as(f64, @floatFromInt(ns_per_embed)) / 1000.0;
    const embeds_per_sec = 1_000_000_000.0 / @as(f64, @floatFromInt(ns_per_embed));

    try stdout.print("  {d} embeddings in {d:.1} ms\n", .{ total_embeds, @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0 });
    try stdout.print("  {d:.1} us/embedding, {d:.0} embeddings/sec\n\n", .{ us_per_embed, embeds_per_sec });

    // Search benchmark
    try stdout.print("Search benchmark:\n", .{});

    const dims = [_]usize{ 256, 512 };
    const doc_counts = [_]usize{ 1000, 10000, 100000 };

    for (dims) |dim| {
        for (doc_counts) |n_docs| {
            try benchSearch(allocator, stdout, dim, n_docs);
        }
    }
}

fn benchSynthetic(allocator: std.mem.Allocator) !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.print("Running synthetic SIMD benchmark:\n", .{});

    const dims = [_]usize{ 256, 512 };
    const doc_counts = [_]usize{ 1000, 10000, 100000 };

    for (dims) |dim| {
        for (doc_counts) |n_docs| {
            try benchSearch(allocator, stdout, dim, n_docs);
        }
    }
}

fn benchSearch(allocator: std.mem.Allocator, stdout: anytype, dim: usize, n_docs: usize) !void {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // Generate random embeddings
    const data = try allocator.alloc(f32, n_docs * dim);
    defer allocator.free(data);
    for (data) |*v| v.* = random.float(f32) * 2.0 - 1.0;

    // Build index
    const data_copy = try allocator.alloc(f32, data.len);
    @memcpy(data_copy, data);

    var flat = zbed.search.FlatIndex.init(allocator, dim);
    defer flat.deinit();
    try flat.build(data_copy, n_docs);

    // Generate random query
    var query: [1024]f32 = undefined;
    for (query[0..dim]) |*v| v.* = random.float(f32) * 2.0 - 1.0;

    // Benchmark search
    const n_iters: usize = 100;
    var results: [10]SearchResult = undefined;

    var timer = try std.time.Timer.start();
    for (0..n_iters) |_| {
        _ = flat.search(query[0..dim], 10, 0.0, &results);
    }
    const elapsed_ns = timer.read();

    const us_per_search = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(n_iters)) / 1000.0;
    try stdout.print("  dim={d:<4} docs={d:<7} {d:>8.1} us/search\n", .{ dim, n_docs, us_per_search });
}

fn cmdStatus(allocator: std.mem.Allocator, args: []const []const u8) !void {
    const stdout = std.io.getStdOut().writer();

    var path: []const u8 = ".";
    if (args.len > 2 and args[2][0] != '-') {
        path = args[2];
    }

    if (!Index.exists(path)) {
        try stdout.print("No index found at {s}/.zbed/index.bin\n", .{path});
        try stdout.print("Run 'zbed index {s}' to create one.\n", .{path});
        return;
    }

    var idx = Index.init(allocator, 1);
    defer idx.deinit();
    try idx.load(path);

    // Count unique files
    var files = std.StringHashMap(void).init(allocator);
    defer files.deinit();
    for (idx.documents.items) |doc| {
        try files.put(doc.file_path, {});
    }

    const emb_bytes = idx.count() * idx.dim * 4;
    const emb_mb = @as(f64, @floatFromInt(emb_bytes)) / (1024.0 * 1024.0);

    try stdout.print("Index: {s}/.zbed/index.bin\n", .{path});
    try stdout.print("  Documents:  {d}\n", .{idx.count()});
    try stdout.print("  Files:      {d}\n", .{files.count()});
    try stdout.print("  Dimensions: {d}\n", .{idx.dim});
    try stdout.print("  Embeddings: {d:.1} MB\n", .{emb_mb});
}
