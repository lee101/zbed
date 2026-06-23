const std = @import("std");
const Allocator = std.mem.Allocator;
const embed_mod = @import("embed.zig");
const gpu_mod = @import("gpu.zig");
const search_mod = @import("search.zig");

const INDEX_MAGIC: u32 = 0x5A424544;
const INDEX_VERSION: u8 = 3;
const INDEX_VERSION_MIN: u8 = 2;

pub const IncrementalStats = struct {
    added: usize = 0,
    updated: usize = 0,
    removed: usize = 0,
    unchanged: usize = 0,
};

pub const MatchLine = struct {
    line_num: u32,
    content: []const u8,
    score: f32,
    kind: DocumentKind,
};

pub const GroupedResult = struct {
    file_path: []const u8,
    best_score: f32,
    matches: []MatchLine,
};

pub const DocumentKind = enum(u8) {
    path = 0,
    text = 1,
    binary = 2,
};

pub const Document = struct {
    file_path: []const u8,
    line_num: u32,
    content: []const u8,
    kind: DocumentKind,
};

pub const WalkOptions = struct {
    search_binaries: bool = false,
    max_file_size: u64 = 10 * 1024 * 1024,
    min_line_length: usize = 3,
    max_line_length: usize = 1200,
    include_path_documents: bool = true,
    gpu_embedder: ?*gpu_mod.GpuEmbedder = null,
};

pub const CountSummary = struct {
    files: usize = 0,
    path_docs: usize = 0,
    text_docs: usize = 0,
    binary_docs: usize = 0,
};

pub const Index = struct {
    documents: std.ArrayListUnmanaged(Document),
    embeddings: std.ArrayListUnmanaged(i8),
    scales: std.ArrayListUnmanaged(f32),
    norms: std.ArrayListUnmanaged(f32),
    file_mtimes: std.StringHashMapUnmanaged(i128),
    dim: usize,
    string_arena: std.heap.ArenaAllocator,
    allocator: Allocator,

    pub fn init(allocator: Allocator, dim: usize) Index {
        return .{
            .documents = .{},
            .embeddings = .{},
            .scales = .{},
            .norms = .{},
            .file_mtimes = .{},
            .dim = dim,
            .string_arena = std.heap.ArenaAllocator.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Index) void {
        self.documents.deinit(self.allocator);
        self.embeddings.deinit(self.allocator);
        self.scales.deinit(self.allocator);
        self.norms.deinit(self.allocator);
        self.file_mtimes.deinit(self.allocator);
        self.string_arena.deinit();
    }

    pub fn reset(self: *Index) void {
        self.documents.deinit(self.allocator);
        self.embeddings.deinit(self.allocator);
        self.scales.deinit(self.allocator);
        self.norms.deinit(self.allocator);
        self.file_mtimes.deinit(self.allocator);
        self.documents = .{};
        self.embeddings = .{};
        self.scales = .{};
        self.norms = .{};
        self.file_mtimes = .{};
        self.string_arena.deinit();
        self.string_arena = std.heap.ArenaAllocator.init(self.allocator);
    }

    pub fn setFileMtime(self: *Index, rel_path: []const u8, mtime: i128) !void {
        const gop = try self.file_mtimes.getOrPut(self.allocator, rel_path);
        if (!gop.found_existing) {
            gop.key_ptr.* = try self.string_arena.allocator().dupe(u8, rel_path);
        }
        gop.value_ptr.* = mtime;
    }

    pub fn addDocumentQuantized(self: *Index, file_path: []const u8, line_num: u32, content: []const u8, kind: DocumentKind, embedding: []const i8, scale: f32, norm: f32) !void {
        const arena = self.string_arena.allocator();
        const owned_path = try arena.dupe(u8, file_path);
        const owned_content = try arena.dupe(u8, content);

        try self.documents.append(self.allocator, .{
            .file_path = owned_path,
            .line_num = line_num,
            .content = owned_content,
            .kind = kind,
        });
        try self.embeddings.appendSlice(self.allocator, embedding[0..self.dim]);
        try self.scales.append(self.allocator, scale);
        try self.norms.append(self.allocator, norm);
    }

    pub fn count(self: *const Index) usize {
        return self.documents.items.len;
    }

    pub fn removeByPath(self: *Index, path: []const u8) usize {
        const single = [_][]const u8{path};
        return self.removeByPaths(&single);
    }

    /// Batch removal: O(n_docs * dim) regardless of how many paths are removed.
    /// Much more efficient than calling removeByPath in a loop.
    pub fn removeByPaths(self: *Index, paths: []const []const u8) usize {
        if (paths.len == 0) return 0;

        var set = std.StringHashMap(void).init(self.allocator);
        defer set.deinit();
        set.ensureTotalCapacity(@intCast(paths.len)) catch return 0;
        for (paths) |p| set.put(p, {}) catch return 0;

        var write: usize = 0;
        for (0..self.documents.items.len) |read| {
            if (set.contains(self.documents.items[read].file_path)) continue;
            if (write != read) {
                self.documents.items[write] = self.documents.items[read];
                self.scales.items[write] = self.scales.items[read];
                self.norms.items[write] = self.norms.items[read];
                @memcpy(
                    self.embeddings.items[write * self.dim ..][0..self.dim],
                    self.embeddings.items[read * self.dim ..][0..self.dim],
                );
            }
            write += 1;
        }
        const removed = self.documents.items.len - write;
        self.documents.items.len = write;
        self.scales.items.len = write;
        self.norms.items.len = write;
        self.embeddings.items.len = write * self.dim;

        for (paths) |p| _ = self.file_mtimes.remove(p);
        return removed;
    }

    /// Reindex a single file: remove old docs then re-embed.
    pub fn reindexFile(
        self: *Index,
        allocator: Allocator,
        root: []const u8,
        rel_path: []const u8,
        model: *const embed_mod.EmbedModel,
        options: WalkOptions,
    ) !usize {
        _ = self.removeByPath(rel_path);

        var full_buf: [4096]u8 = undefined;
        const full_path = try std.fmt.bufPrint(&full_buf, "{s}/{s}", .{ root, rel_path });

        const stat = std.fs.cwd().statFile(full_path) catch return 0; // file deleted
        var scratch = embed_mod.EmbedScratch{};
        var quantized = embed_mod.QuantizedEmbedding{};
        const before = self.count();
        try processFile(allocator, full_path, rel_path, model, self, options, &scratch, &quantized);
        try self.setFileMtime(rel_path, stat.mtime);
        return self.count() - before;
    }

    pub fn buildSearchIndex(self: *const Index) search_mod.QuantizedFlatIndex {
        return search_mod.QuantizedFlatIndex.init(self.embeddings.items, self.norms.items, self.dim);
    }

    /// Group raw search results by file path. Returned slice is owned by `arena`.
    /// `results` should be sorted by score descending (the search() function does this).
    pub fn groupResults(
        self: *const Index,
        arena: Allocator,
        results: []const search_mod.SearchResult,
        max_groups: usize,
        max_lines_per_group: usize,
    ) ![]GroupedResult {
        if (results.len == 0 or max_groups == 0) return &.{};

        // Bucket by file_path, preserving first-seen order (which is best-score order)
        var groups: std.ArrayListUnmanaged(GroupedResult) = .{};
        var lines_lists: std.ArrayListUnmanaged(std.ArrayListUnmanaged(MatchLine)) = .{};
        var by_path = std.StringHashMap(usize).init(arena);
        defer by_path.deinit();

        for (results) |r| {
            const doc = self.documents.items[r.doc_idx];
            const gop = try by_path.getOrPut(doc.file_path);
            if (!gop.found_existing) {
                if (groups.items.len >= max_groups) {
                    // Drop this match -- group cap hit
                    _ = by_path.remove(doc.file_path);
                    continue;
                }
                gop.value_ptr.* = groups.items.len;
                try groups.append(arena, .{
                    .file_path = doc.file_path,
                    .best_score = r.score,
                    .matches = &.{},
                });
                try lines_lists.append(arena, .{});
            }
            const idx = gop.value_ptr.*;
            const lines = &lines_lists.items[idx];
            if (lines.items.len >= max_lines_per_group) continue;
            try lines.append(arena, .{
                .line_num = doc.line_num,
                .content = doc.content,
                .score = r.score,
                .kind = doc.kind,
            });
        }

        // Materialize line slices into the GroupedResult array
        for (groups.items, 0..) |*g, i| {
            g.matches = try lines_lists.items[i].toOwnedSlice(arena);
        }
        return groups.toOwnedSlice(arena);
    }

    pub fn save(self: *const Index, dir_path: []const u8) !void {
        var zbed_buf: [4096]u8 = undefined;
        const zbed_path = try std.fmt.bufPrint(&zbed_buf, "{s}/.zbed", .{dir_path});
        std.fs.cwd().makeDir(zbed_path) catch |err| {
            if (err != error.PathAlreadyExists) return err;
        };

        var path_buf: [4096]u8 = undefined;
        const full_path = try std.fmt.bufPrint(&path_buf, "{s}/.zbed/index.bin", .{dir_path});
        const file = try std.fs.cwd().createFile(full_path, .{});
        defer file.close();

        try writeU32(file, INDEX_MAGIC);
        try file.writeAll(&[_]u8{INDEX_VERSION});
        try writeU32(file, @intCast(self.dim));
        try writeU32(file, @intCast(self.count()));

        for (self.documents.items, 0..) |doc, idx| {
            try file.writeAll(&[_]u8{@intFromEnum(doc.kind)});
            try writeU32(file, doc.line_num);
            try writeString(file, doc.file_path);
            try writeString(file, doc.content);
            try writeF32(file, self.scales.items[idx]);
            try writeF32(file, self.norms.items[idx]);
        }

        try file.writeAll(std.mem.sliceAsBytes(self.embeddings.items));

        // v3: file mtimes section
        try writeU32(file, @intCast(self.file_mtimes.count()));
        var it = self.file_mtimes.iterator();
        while (it.next()) |entry| {
            try writeString(file, entry.key_ptr.*);
            try writeI128(file, entry.value_ptr.*);
        }
    }

    pub fn load(self: *Index, dir_path: []const u8) !void {
        self.reset();

        var path_buf: [4096]u8 = undefined;
        const full_path = try std.fmt.bufPrint(&path_buf, "{s}/.zbed/index.bin", .{dir_path});
        const file = try std.fs.cwd().openFile(full_path, .{});
        defer file.close();

        const magic = try readU32(file);
        if (magic != INDEX_MAGIC) return error.InvalidMagic;

        var version_buf: [1]u8 = undefined;
        const version_read = try file.readAll(&version_buf);
        if (version_read != 1) return error.UnexpectedEof;
        const version = version_buf[0];
        if (version < INDEX_VERSION_MIN or version > INDEX_VERSION) return error.UnsupportedVersion;

        self.dim = try readU32(file);
        const n_docs = try readU32(file);
        const arena = self.string_arena.allocator();

        try self.documents.ensureTotalCapacity(self.allocator, n_docs);
        try self.scales.ensureTotalCapacity(self.allocator, n_docs);
        try self.norms.ensureTotalCapacity(self.allocator, n_docs);

        for (0..n_docs) |_| {
            var kind_buf: [1]u8 = undefined;
            const kind_read = try file.readAll(&kind_buf);
            if (kind_read != 1) return error.UnexpectedEof;
            const line_num = try readU32(file);
            const file_path = try readString(file, arena);
            const content = try readString(file, arena);
            const scale = try readF32(file);
            const norm = try readF32(file);

            const kind = std.meta.intToEnum(DocumentKind, kind_buf[0]) catch return error.InvalidDocumentKind;
            try self.documents.append(self.allocator, .{
                .file_path = file_path,
                .line_num = line_num,
                .content = content,
                .kind = kind,
            });
            try self.scales.append(self.allocator, scale);
            try self.norms.append(self.allocator, norm);
        }

        const emb_count = @as(usize, n_docs) * self.dim;
        try self.embeddings.resize(self.allocator, emb_count);
        const emb_bytes = std.mem.sliceAsBytes(self.embeddings.items);
        const emb_read = try file.readAll(emb_bytes);
        if (emb_read != emb_bytes.len) return error.UnexpectedEof;

        // v3: file mtimes section
        if (version >= 3) {
            const n_files = readU32(file) catch 0;
            try self.file_mtimes.ensureTotalCapacity(self.allocator, n_files);
            for (0..n_files) |_| {
                const path = try readString(file, arena);
                const mtime = try readI128(file);
                self.file_mtimes.putAssumeCapacity(path, mtime);
            }
        }
    }

    pub fn exists(dir_path: []const u8) bool {
        var path_buf: [4096]u8 = undefined;
        const full_path = std.fmt.bufPrint(&path_buf, "{s}/.zbed/index.bin", .{dir_path}) catch return false;
        std.fs.cwd().access(full_path, .{}) catch return false;
        return true;
    }

    pub fn summarize(self: *const Index, allocator: Allocator) !CountSummary {
        var files = std.StringHashMap(void).init(allocator);
        defer files.deinit();

        var summary = CountSummary{};
        for (self.documents.items) |doc| {
            try files.put(doc.file_path, {});
            switch (doc.kind) {
                .path => summary.path_docs += 1,
                .text => summary.text_docs += 1,
                .binary => summary.binary_docs += 1,
            }
        }
        summary.files = files.count();
        return summary;
    }
};

fn writeU32(file: std.fs.File, val: u32) !void {
    const bytes = std.mem.toBytes(std.mem.nativeToLittle(u32, val));
    try file.writeAll(&bytes);
}

fn readU32(file: std.fs.File) !u32 {
    var buf: [4]u8 = undefined;
    const n = try file.readAll(&buf);
    if (n != 4) return error.UnexpectedEof;
    return std.mem.littleToNative(u32, std.mem.bytesToValue(u32, &buf));
}

fn writeF32(file: std.fs.File, val: f32) !void {
    try writeU32(file, @bitCast(val));
}

fn readF32(file: std.fs.File) !f32 {
    const bits = try readU32(file);
    return @bitCast(bits);
}

fn writeI128(file: std.fs.File, val: i128) !void {
    const bytes = std.mem.toBytes(std.mem.nativeToLittle(i128, val));
    try file.writeAll(&bytes);
}

fn readI128(file: std.fs.File) !i128 {
    var buf: [16]u8 = undefined;
    const n = try file.readAll(&buf);
    if (n != 16) return error.UnexpectedEof;
    return std.mem.littleToNative(i128, std.mem.bytesToValue(i128, &buf));
}

fn writeString(file: std.fs.File, s: []const u8) !void {
    try writeU32(file, @intCast(s.len));
    try file.writeAll(s);
}

fn readString(file: std.fs.File, allocator: Allocator) ![]const u8 {
    const len = try readU32(file);
    if (len > 10 * 1024 * 1024) return error.StringTooLong;
    const buf = try allocator.alloc(u8, len);
    const n = try file.readAll(buf);
    if (n != len) return error.UnexpectedEof;
    return buf;
}

const text_ext_map = std.StaticStringMapWithEql(void, std.static_string_map.eqlAsciiIgnoreCase).initComptime(.{
    .{ ".txt", {} },     .{ ".md", {} },         .{ ".rst", {} },  .{ ".tex", {} },
    .{ ".go", {} },      .{ ".py", {} },         .{ ".js", {} },   .{ ".ts", {} },
    .{ ".jsx", {} },     .{ ".tsx", {} },        .{ ".c", {} },    .{ ".cpp", {} },
    .{ ".cc", {} },      .{ ".cxx", {} },        .{ ".h", {} },    .{ ".hpp", {} },
    .{ ".hh", {} },      .{ ".rs", {} },         .{ ".rb", {} },   .{ ".php", {} },
    .{ ".java", {} },    .{ ".cs", {} },         .{ ".swift", {} }, .{ ".kt", {} },
    .{ ".scala", {} },   .{ ".zig", {} },        .{ ".lua", {} },  .{ ".json", {} },
    .{ ".jsonc", {} },   .{ ".yaml", {} },       .{ ".yml", {} },  .{ ".toml", {} },
    .{ ".ini", {} },     .{ ".conf", {} },       .{ ".cfg", {} },  .{ ".sh", {} },
    .{ ".bash", {} },    .{ ".zsh", {} },        .{ ".fish", {} }, .{ ".ps1", {} },
    .{ ".bat", {} },     .{ ".cmd", {} },        .{ ".sql", {} },  .{ ".graphql", {} },
    .{ ".proto", {} },   .{ ".html", {} },       .{ ".htm", {} },  .{ ".css", {} },
    .{ ".scss", {} },    .{ ".sass", {} },       .{ ".less", {} }, .{ ".xml", {} },
    .{ ".dockerfile", {} }, .{ ".gitignore", {} }, .{ ".vue", {} },  .{ ".svelte", {} },
    .{ ".elm", {} },     .{ ".clj", {} },        .{ ".cljs", {} }, .{ ".ex", {} },
    .{ ".exs", {} },     .{ ".erl", {} },        .{ ".hs", {} },   .{ ".ml", {} },
    .{ ".nim", {} },     .{ ".d", {} },          .{ ".dart", {} }, .{ ".r", {} },
    .{ ".jl", {} },      .{ ".pl", {} },         .{ ".pm", {} },   .{ ".tcl", {} },
});

const binary_ext_map = std.StaticStringMapWithEql(void, std.static_string_map.eqlAsciiIgnoreCase).initComptime(.{
    .{ ".exe", {} },  .{ ".dll", {} },   .{ ".so", {} },    .{ ".dylib", {} },
    .{ ".o", {} },    .{ ".a", {} },     .{ ".obj", {} },   .{ ".lib", {} },
    .{ ".zip", {} },  .{ ".tar", {} },   .{ ".gz", {} },    .{ ".7z", {} },
    .{ ".rar", {} },  .{ ".bz2", {} },   .{ ".xz", {} },    .{ ".zst", {} },
    .{ ".jpg", {} },  .{ ".jpeg", {} },  .{ ".png", {} },   .{ ".gif", {} },
    .{ ".bmp", {} },  .{ ".webp", {} },  .{ ".ico", {} },   .{ ".tiff", {} },
    .{ ".pdf", {} },  .{ ".mp3", {} },   .{ ".opus", {} },  .{ ".flac", {} },
    .{ ".wav", {} },  .{ ".m4a", {} },   .{ ".aac", {} },   .{ ".ogg", {} },
    .{ ".mp4", {} },  .{ ".m4v", {} },   .{ ".avi", {} },   .{ ".mov", {} },
    .{ ".mkv", {} },  .{ ".webm", {} },  .{ ".ttf", {} },   .{ ".otf", {} },
    .{ ".woff", {} }, .{ ".woff2", {} }, .{ ".db", {} },    .{ ".sqlite", {} },
    .{ ".sqlite3", {} }, .{ ".bin", {} }, .{ ".class", {} }, .{ ".jar", {} },
    .{ ".pyc", {} },  .{ ".wasm", {} },
});

const DEFAULT_IGNORES = [_][]const u8{
    ".git", ".zbed", ".bed", "node_modules", "vendor", "dist", "build", "target",
    ".cache", "__pycache__", ".venv", "model",
};

const FileType = enum { text, binary, unknown };

pub fn walkAndIndex(allocator: Allocator, root: []const u8, model: *const embed_mod.EmbedModel, index: *Index, options: WalkOptions, progress_fn: ?*const fn (usize) void) !void {
    var task_arena = std.heap.ArenaAllocator.init(allocator);
    defer task_arena.deinit();

    var tasks: std.ArrayListUnmanaged(FileTask) = .{};
    defer tasks.deinit(allocator);
    try collectAllFiles(allocator, task_arena.allocator(), root, &tasks);
    if (tasks.items.len == 0) return;

    // GPU embedder is not thread-safe (single CUDA stream); use serial path.
    if (options.gpu_embedder != null) {
        var scratch = embed_mod.EmbedScratch{};
        var quantized = embed_mod.QuantizedEmbedding{};
        for (tasks.items) |t| {
            try processFile(allocator, t.full_path, t.rel_path, model, index, options, &scratch, &quantized);
            try index.setFileMtime(t.rel_path, t.mtime);
            if (progress_fn) |cb| cb(index.count());
        }
        return;
    }

    try embedTasksParallel(allocator, model, index, options, tasks.items, progress_fn);
}

const FileTask = struct {
    full_path: []const u8,
    rel_path: []const u8,
    mtime: i128 = 0,
};

const WorkerCtx = struct {
    allocator: Allocator,
    model: *const embed_mod.EmbedModel,
    options: WalkOptions,
    indices: []Index,
    progress_count: *std.atomic.Value(usize),
    progress_fn: ?*const fn (usize) void,
};

fn workerProcessFile(thread_id: usize, task: FileTask, ctx: *WorkerCtx) void {
    var scratch = embed_mod.EmbedScratch{};
    var quantized = embed_mod.QuantizedEmbedding{};
    const idx = &ctx.indices[thread_id];

    processFile(ctx.allocator, task.full_path, task.rel_path, ctx.model, idx, ctx.options, &scratch, &quantized) catch {};

    // Throttle the callback to avoid stderr garbling and contention.
    const c = ctx.progress_count.fetchAdd(1, .monotonic) + 1;
    if (c % 256 == 0) if (ctx.progress_fn) |cb| cb(c);
}

/// Walk the filesystem, applying default + nested .gitignore filters, and collect all files.
fn collectAllFiles(allocator: Allocator, arena: Allocator, root: []const u8, tasks: *std.ArrayListUnmanaged(FileTask)) !void {
    var ignore = IgnoreFilter.init(allocator);
    defer ignore.deinit();
    for (DEFAULT_IGNORES) |pattern| try ignore.add(pattern, false, true);

    var gi_buf: [4096]u8 = undefined;
    if (std.fmt.bufPrint(&gi_buf, "{s}/.gitignore", .{root})) |gi_path| {
        ignore.loadFile(gi_path) catch {};
    } else |_| {}

    var dir = try std.fs.cwd().openDir(root, .{ .iterate = true });
    defer dir.close();
    try collectFiles(allocator, arena, dir, root, "", &ignore, tasks);
}

/// Spawn a worker pool, process the given tasks in parallel, and merge results into `index`.
fn embedTasksParallel(
    allocator: Allocator,
    model: *const embed_mod.EmbedModel,
    index: *Index,
    options: WalkOptions,
    tasks: []const FileTask,
    progress_fn: ?*const fn (usize) void,
) !void {
    if (tasks.len == 0) return;

    const cpu_count = std.Thread.getCpuCount() catch 4;
    const n_workers = @max(@min(cpu_count, 12), 1);

    var pool: std.Thread.Pool = undefined;
    try pool.init(.{ .allocator = allocator, .n_jobs = n_workers, .track_ids = true });
    defer pool.deinit();

    const id_count = pool.getIdCount();
    const indices = try allocator.alloc(Index, id_count);
    defer allocator.free(indices);
    for (indices) |*i| i.* = Index.init(allocator, model.embed_dim);
    defer for (indices) |*i| i.deinit();

    var progress_count = std.atomic.Value(usize).init(0);
    var ctx = WorkerCtx{
        .allocator = allocator,
        .model = model,
        .options = options,
        .indices = indices,
        .progress_count = &progress_count,
        .progress_fn = progress_fn,
    };

    var wg: std.Thread.WaitGroup = .{};
    for (tasks) |task| pool.spawnWgId(&wg, workerProcessFile, .{ task, &ctx });
    pool.waitAndWork(&wg);

    // Pre-size before merge to avoid exponential reallocation
    var total: usize = 0;
    for (indices) |*pt| total += pt.documents.items.len;
    try index.documents.ensureTotalCapacity(index.allocator, index.documents.items.len + total);
    try index.scales.ensureTotalCapacity(index.allocator, index.scales.items.len + total);
    try index.norms.ensureTotalCapacity(index.allocator, index.norms.items.len + total);
    try index.embeddings.ensureTotalCapacity(index.allocator, index.embeddings.items.len + total * index.dim);
    try index.file_mtimes.ensureTotalCapacity(index.allocator, @intCast(index.file_mtimes.count() + tasks.len));

    for (indices) |*per_thread_idx| {
        for (per_thread_idx.documents.items, 0..) |doc, i| {
            const emb = per_thread_idx.embeddings.items[i * index.dim .. (i + 1) * index.dim];
            try index.addDocumentQuantized(doc.file_path, doc.line_num, doc.content, doc.kind, emb, per_thread_idx.scales.items[i], per_thread_idx.norms.items[i]);
        }
    }

    for (tasks) |task| try index.setFileMtime(task.rel_path, task.mtime);
}

pub fn walkAndIndexIncremental(allocator: Allocator, root: []const u8, model: *const embed_mod.EmbedModel, index: *Index, options: WalkOptions, progress_fn: ?*const fn (usize) void) !IncrementalStats {
    var stats = IncrementalStats{};

    // If we have no mtime info, fall back to a full build
    if (index.file_mtimes.count() == 0) {
        index.reset();
        try walkAndIndex(allocator, root, model, index, options, progress_fn);
        stats.added = index.file_mtimes.count();
        return stats;
    }

    // GPU path: full reset+rebuild (incremental complexity not worth it)
    if (options.gpu_embedder != null) {
        index.reset();
        try walkAndIndex(allocator, root, model, index, options, progress_fn);
        return stats;
    }

    // Phase 1: walk filesystem
    var task_arena = std.heap.ArenaAllocator.init(allocator);
    defer task_arena.deinit();

    var tasks: std.ArrayListUnmanaged(FileTask) = .{};
    defer tasks.deinit(allocator);
    try collectAllFiles(allocator, task_arena.allocator(), root, &tasks);

    // Phase 2: diff against existing mtimes
    var seen = std.StringHashMap(void).init(allocator);
    defer seen.deinit();
    try seen.ensureTotalCapacity(@intCast(tasks.items.len));

    var to_process: std.ArrayListUnmanaged(FileTask) = .{};
    defer to_process.deinit(allocator);
    var to_remove: std.ArrayListUnmanaged([]const u8) = .{};
    defer to_remove.deinit(allocator);

    for (tasks.items) |t| {
        seen.putAssumeCapacity(t.rel_path, {});
        if (index.file_mtimes.get(t.rel_path)) |old_mtime| {
            if (old_mtime == t.mtime) {
                stats.unchanged += 1;
                continue;
            }
            try to_remove.append(allocator, t.rel_path);
            stats.updated += 1;
        } else {
            stats.added += 1;
        }
        try to_process.append(allocator, t);
    }

    // Phase 3: find deleted files (in mtimes but no longer on disk)
    var mit = index.file_mtimes.iterator();
    while (mit.next()) |entry| {
        if (!seen.contains(entry.key_ptr.*)) {
            try to_remove.append(allocator, entry.key_ptr.*);
            stats.removed += 1;
        }
    }

    // Phase 4: single batched removal (avoids quadratic compaction)
    _ = index.removeByPaths(to_remove.items);

    // Phase 5: parallel re-embed of new + changed files
    try embedTasksParallel(allocator, model, index, options, to_process.items, progress_fn);

    return stats;
}

fn collectFiles(
    allocator: Allocator,
    arena: Allocator,
    dir: std.fs.Dir,
    root: []const u8,
    rel_prefix: []const u8,
    ignore: *const IgnoreFilter,
    tasks: *std.ArrayListUnmanaged(FileTask),
) !void {
    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        var rel_buf: [4096]u8 = undefined;
        const rel_path = if (rel_prefix.len > 0)
            try std.fmt.bufPrint(&rel_buf, "{s}/{s}", .{ rel_prefix, entry.name })
        else
            try std.fmt.bufPrint(&rel_buf, "{s}", .{entry.name});

        switch (entry.kind) {
            .directory => {
                if (ignore.shouldIgnore(rel_path, true)) continue;
                var subdir = dir.openDir(entry.name, .{ .iterate = true }) catch continue;
                defer subdir.close();

                // Lazy: only allocate child filter if a .gitignore actually exists.
                const has_gi = blk: {
                    subdir.access(".gitignore", .{}) catch break :blk false;
                    break :blk true;
                };
                var child_ignore: ?IgnoreFilter = null;
                defer if (child_ignore) |*c| c.deinit();
                const next_ignore: *const IgnoreFilter = if (has_gi) child: {
                    child_ignore = IgnoreFilter.initWithParent(allocator, ignore, rel_path);
                    var gi_buf: [4096]u8 = undefined;
                    if (std.fmt.bufPrint(&gi_buf, "{s}/{s}/.gitignore", .{ root, rel_path })) |gi_path| {
                        child_ignore.?.loadFile(gi_path) catch {};
                    } else |_| {}
                    break :child &child_ignore.?;
                } else ignore;

                try collectFiles(allocator, arena, subdir, root, rel_path, next_ignore, tasks);
            },
            .file => {
                if (ignore.shouldIgnore(rel_path, false)) continue;
                const stat = dir.statFile(entry.name) catch continue;
                const owned_rel = try arena.dupe(u8, rel_path);
                const full_path = try std.fmt.allocPrint(arena, "{s}/{s}", .{ root, rel_path });
                try tasks.append(allocator, .{
                    .full_path = full_path,
                    .rel_path = owned_rel,
                    .mtime = stat.mtime,
                });
            },
            else => {},
        }
    }
}

fn classifyByExtension(rel_path: []const u8) FileType {
    if (isTextFile(rel_path)) return .text;
    if (isBinaryExtension(rel_path)) return .binary;
    return .unknown;
}

fn embedNameDoc(allocator: Allocator, rel_path: []const u8, kind: DocumentKind, model: *const embed_mod.EmbedModel, index: *Index, options: WalkOptions, scratch: *embed_mod.EmbedScratch, quantized: *embed_mod.QuantizedEmbedding) !void {
    const display_name = std.fs.path.basename(rel_path);
    const search_text = try normalizePathForSearch(allocator, rel_path);
    defer allocator.free(search_text);

    const valid = if (options.gpu_embedder) |gpu_embedder|
        try gpu_embedder.embedQuantized(model, search_text, scratch, quantized)
    else
        model.embedQuantizedWithScratch(search_text, scratch, quantized);
    if (valid == 0) return;

    try index.addDocumentQuantized(rel_path, 0, display_name, kind, quantized.data[0..model.embed_dim], quantized.scale, quantized.norm);
}

fn embedTextLines(data: []const u8, rel_path: []const u8, model: *const embed_mod.EmbedModel, index: *Index, options: WalkOptions, scratch: *embed_mod.EmbedScratch, quantized: *embed_mod.QuantizedEmbedding) !void {
    var lines = std.mem.splitScalar(u8, data, '\n');
    var line_num: u32 = 0;
    while (lines.next()) |line| {
        line_num += 1;
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len < options.min_line_length) continue;
        if (trimmed.len > options.max_line_length) continue;
        const valid = if (options.gpu_embedder) |gpu_embedder|
            try gpu_embedder.embedQuantized(model, trimmed, scratch, quantized)
        else
            model.embedQuantizedWithScratch(trimmed, scratch, quantized);
        if (valid == 0) continue;

        try index.addDocumentQuantized(rel_path, line_num, trimmed, .text, quantized.data[0..model.embed_dim], quantized.scale, quantized.norm);
    }
}

/// Process a single file: open once, classify, and index. Used by both serial and parallel walks.
fn processFile(allocator: Allocator, full_path: []const u8, rel_path: []const u8, model: *const embed_mod.EmbedModel, index: *Index, options: WalkOptions, scratch: *embed_mod.EmbedScratch, quantized: *embed_mod.QuantizedEmbedding) !void {
    const ext_class = classifyByExtension(rel_path);

    if (ext_class == .binary) {
        if (options.search_binaries) try embedNameDoc(allocator, rel_path, .binary, model, index, options, scratch, quantized);
        return;
    }

    // Need file contents for both .text and .unknown
    const file = std.fs.cwd().openFile(full_path, .{}) catch return;
    defer file.close();

    // readToEndAlloc enforces max_file_size; no extra stat needed
    const data = file.readToEndAlloc(allocator, options.max_file_size) catch return;
    defer allocator.free(data);

    if (ext_class == .unknown and isDataBinary(data)) {
        if (options.search_binaries) try embedNameDoc(allocator, rel_path, .binary, model, index, options, scratch, quantized);
        return;
    }

    if (options.include_path_documents) {
        try embedNameDoc(allocator, rel_path, .path, model, index, options, scratch, quantized);
    }
    try embedTextLines(data, rel_path, model, index, options, scratch, quantized);
}

pub fn isTextFile(path: []const u8) bool {
    const basename = std.fs.path.basename(path);
    const known_names = [_][]const u8{
        "Makefile", "Dockerfile", "README", "LICENSE", "build.zig", "build.zig.zon",
    };
    for (known_names) |name| {
        if (std.mem.eql(u8, basename, name)) return true;
    }
    const ext = std.fs.path.extension(path);
    if (ext.len == 0) return false;
    return text_ext_map.has(ext);
}

fn isBinaryExtension(path: []const u8) bool {
    const ext = std.fs.path.extension(path);
    if (ext.len == 0) return false;
    return binary_ext_map.has(ext);
}

fn isDataBinary(data: []const u8) bool {
    const n = @min(data.len, 8192);
    if (n == 0) return false;
    const buf = data[0..n];
    if (std.mem.indexOfScalar(u8, buf, 0) != null) return true;

    var non_printable: usize = 0;
    for (buf) |byte| {
        if ((byte < 32 and byte != '\n' and byte != '\r' and byte != '\t') or (byte > 126 and byte < 128)) {
            non_printable += 1;
        }
    }
    return @as(f32, @floatFromInt(non_printable)) / @as(f32, @floatFromInt(n)) > 0.3;
}

fn normalizePathForSearch(allocator: Allocator, rel_path: []const u8) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .{};
    errdefer out.deinit(allocator);

    var last_space = false;
    for (rel_path) |byte| {
        const lowered = std.ascii.toLower(byte);
        const mapped = switch (lowered) {
            '/', '\\', '.', '-', '_' => ' ',
            else => lowered,
        };

        if (mapped == ' ') {
            if (last_space) continue;
            last_space = true;
        } else {
            last_space = false;
        }
        try out.append(allocator, mapped);
    }

    return out.toOwnedSlice(allocator);
}

const IgnoreFilter = struct {
    patterns: std.ArrayListUnmanaged(IgnorePattern),
    allocator: Allocator,
    parent: ?*const IgnoreFilter,
    /// Path prefix this filter was loaded at (relative to root). Patterns from
    /// nested .gitignore files match against paths relative to this prefix.
    base_prefix: []const u8,

    const IgnorePattern = struct {
        pattern: []const u8,
        negation: bool,
        dir_only: bool,
    };

    fn init(allocator: Allocator) IgnoreFilter {
        return .{
            .patterns = .{},
            .allocator = allocator,
            .parent = null,
            .base_prefix = "",
        };
    }

    fn initWithParent(allocator: Allocator, parent: *const IgnoreFilter, base_prefix: []const u8) IgnoreFilter {
        return .{
            .patterns = .{},
            .allocator = allocator,
            .parent = parent,
            .base_prefix = base_prefix,
        };
    }

    fn deinit(self: *IgnoreFilter) void {
        for (self.patterns.items) |pattern| self.allocator.free(pattern.pattern);
        self.patterns.deinit(self.allocator);
    }

    fn add(self: *IgnoreFilter, pattern: []const u8, negation: bool, dir_only: bool) !void {
        const owned = try self.allocator.dupe(u8, pattern);
        try self.patterns.append(self.allocator, .{
            .pattern = owned,
            .negation = negation,
            .dir_only = dir_only,
        });
    }

    fn loadFile(self: *IgnoreFilter, path: []const u8) !void {
        const file = std.fs.cwd().openFile(path, .{}) catch return;
        defer file.close();

        const data = try file.readToEndAlloc(self.allocator, 1024 * 1024);
        defer self.allocator.free(data);

        var lines = std.mem.splitScalar(u8, data, '\n');
        while (lines.next()) |raw_line| {
            var line = std.mem.trim(u8, raw_line, " \t\r");
            if (line.len == 0 or line[0] == '#') continue;

            var negation = false;
            if (line[0] == '!') {
                negation = true;
                line = line[1..];
            }

            var dir_only = false;
            if (line.len > 0 and line[line.len - 1] == '/') {
                dir_only = true;
                line = line[0 .. line.len - 1];
            }

            if (line.len > 0 and line[0] == '/') line = line[1..];
            if (line.len == 0) continue;
            try self.add(line, negation, dir_only);
        }
    }

    fn shouldIgnore(self: *const IgnoreFilter, rel_path: []const u8, is_dir: bool) bool {
        // Parent rules first (so children can override)
        var ignored = if (self.parent) |p| p.shouldIgnore(rel_path, is_dir) else false;

        // Strip our base prefix so patterns match relative to where the .gitignore lives.
        // Require an exact path-component boundary so "src" doesn't match "src2/foo".
        const bp = self.base_prefix;
        const matches_prefix = bp.len > 0 and
            std.mem.startsWith(u8, rel_path, bp) and
            (rel_path.len == bp.len or rel_path[bp.len] == '/');
        const sub_path = if (matches_prefix)
            std.mem.trimLeft(u8, rel_path[bp.len..], "/")
        else
            rel_path;

        for (self.patterns.items) |pattern| {
            if (pattern.dir_only and !is_dir) {
                if (!matchesDirectoryPattern(pattern.pattern, sub_path)) continue;
            } else if (!(matchGlob(pattern.pattern, sub_path) or matchBasename(pattern.pattern, sub_path))) {
                continue;
            }
            ignored = !pattern.negation;
        }
        return ignored;
    }
};

fn matchesDirectoryPattern(pattern: []const u8, path: []const u8) bool {
    if (matchBasename(pattern, path)) return true;
    return std.mem.startsWith(u8, path, pattern) or std.mem.indexOf(u8, path, pattern) != null;
}

fn matchGlob(pattern: []const u8, path: []const u8) bool {
    if (std.mem.eql(u8, pattern, path)) return true;
    if (std.mem.indexOfScalar(u8, pattern, '/') == null) return matchSimpleGlob(pattern, std.fs.path.basename(path));
    return matchSimpleGlob(pattern, path);
}

fn matchBasename(pattern: []const u8, path: []const u8) bool {
    return matchSimpleGlob(pattern, std.fs.path.basename(path));
}

fn matchSimpleGlob(pattern: []const u8, str: []const u8) bool {
    var pi: usize = 0;
    var si: usize = 0;
    var star_p: ?usize = null;
    var star_s: usize = 0;

    while (si < str.len) {
        if (pi < pattern.len and (pattern[pi] == str[si] or pattern[pi] == '?')) {
            pi += 1;
            si += 1;
        } else if (pi < pattern.len and pattern[pi] == '*') {
            star_p = pi + 1;
            star_s = si;
            pi += 1;
        } else if (star_p) |saved| {
            pi = saved;
            star_s += 1;
            si = star_s;
        } else {
            return false;
        }
    }

    while (pi < pattern.len and pattern[pi] == '*') pi += 1;
    return pi == pattern.len;
}

test "index save and load round-trip" {
    const allocator = std.testing.allocator;
    var idx = Index.init(allocator, 4);
    defer idx.deinit();

    const emb1 = [_]i8{ 1, 2, 3, 4 };
    const emb2 = [_]i8{ 5, 6, 7, 8 };
    try idx.addDocumentQuantized("test.txt", 0, "test.txt", .path, &emb1, 0.1, search_mod.quantizedNorm(&emb1));
    try idx.addDocumentQuantized("test.txt", 2, "foo bar", .text, &emb2, 0.2, search_mod.quantizedNorm(&emb2));

    const tmp_dir = "/tmp/zbed_test_idx";
    std.fs.cwd().makeDir(tmp_dir) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir) catch {};

    try idx.save(tmp_dir);

    var idx2 = Index.init(allocator, 1);
    defer idx2.deinit();
    try idx2.load(tmp_dir);

    try std.testing.expectEqual(@as(usize, 2), idx2.count());
    try std.testing.expectEqual(@as(usize, 4), idx2.dim);
    try std.testing.expectEqual(DocumentKind.path, idx2.documents.items[0].kind);
    try std.testing.expectEqual(DocumentKind.text, idx2.documents.items[1].kind);
    try std.testing.expectEqual(@as(i8, 8), idx2.embeddings.items[7]);
}

test "normalize path for search" {
    const allocator = std.testing.allocator;
    const normalized = try normalizePathForSearch(allocator, "audio/My-File.opus");
    defer allocator.free(normalized);
    try std.testing.expectEqualStrings("audio my file opus", normalized);
}

test "text file detection" {
    try std.testing.expect(isTextFile("main.go"));
    try std.testing.expect(!isTextFile("song.mp3"));
    try std.testing.expect(isBinaryExtension("movie.mp4"));
}
