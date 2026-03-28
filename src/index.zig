const std = @import("std");
const Allocator = std.mem.Allocator;
const embed_mod = @import("embed.zig");
const gpu_mod = @import("gpu.zig");
const search_mod = @import("search.zig");

const INDEX_MAGIC: u32 = 0x5A424544;
const INDEX_VERSION: u8 = 2;

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
    dim: usize,
    string_arena: std.heap.ArenaAllocator,
    allocator: Allocator,

    pub fn init(allocator: Allocator, dim: usize) Index {
        return .{
            .documents = .{},
            .embeddings = .{},
            .scales = .{},
            .norms = .{},
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
        self.string_arena.deinit();
    }

    pub fn reset(self: *Index) void {
        self.documents.deinit(self.allocator);
        self.embeddings.deinit(self.allocator);
        self.scales.deinit(self.allocator);
        self.norms.deinit(self.allocator);
        self.documents = .{};
        self.embeddings = .{};
        self.scales = .{};
        self.norms = .{};
        self.string_arena.deinit();
        self.string_arena = std.heap.ArenaAllocator.init(self.allocator);
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

    /// Remove all documents with the given file_path. Returns number removed.
    pub fn removeByPath(self: *Index, path: []const u8) usize {
        var removed: usize = 0;
        var i: usize = 0;
        while (i < self.documents.items.len) {
            if (std.mem.eql(u8, self.documents.items[i].file_path, path)) {
                _ = self.documents.orderedRemove(i);
                _ = self.scales.orderedRemove(i);
                _ = self.norms.orderedRemove(i);
                // Remove dim-sized embedding slice
                const start = i * self.dim;
                const end = start + self.dim;
                if (end <= self.embeddings.items.len) {
                    std.mem.copyForwards(i8, self.embeddings.items[start..], self.embeddings.items[end..]);
                    self.embeddings.items.len -= self.dim;
                }
                removed += 1;
            } else {
                i += 1;
            }
        }
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

        std.fs.cwd().access(full_path, .{}) catch return 0; // file deleted

        const file_type = detectFileType(full_path, rel_path, options);
        var scratch = embed_mod.EmbedScratch{};
        var quantized = embed_mod.QuantizedEmbedding{};

        const before = self.count();
        switch (file_type) {
            .ignored, .too_large => {},
            .binary => {
                if (options.search_binaries) {
                    try indexBinaryFile(allocator, rel_path, model, self, options, &scratch, &quantized);
                }
            },
            .text => {
                try indexTextFile(allocator, full_path, rel_path, model, self, options, &scratch, &quantized);
            },
        }
        return self.count() - before;
    }

    pub fn buildSearchIndex(self: *const Index) search_mod.QuantizedFlatIndex {
        return search_mod.QuantizedFlatIndex.init(self.embeddings.items, self.norms.items, self.dim);
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
        if (version_buf[0] != INDEX_VERSION) return error.UnsupportedVersion;

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

const TEXT_EXTENSIONS = [_][]const u8{
    ".txt", ".md", ".rst", ".tex", ".go", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".c", ".cpp", ".h", ".hpp", ".rs", ".rb", ".php", ".java", ".cs", ".swift",
    ".kt", ".scala", ".zig", ".lua", ".json", ".yaml", ".yml", ".toml", ".ini",
    ".conf", ".cfg", ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
    ".sql", ".graphql", ".proto", ".html", ".css", ".scss", ".sass", ".less",
    ".xml", ".dockerfile", ".gitignore",
};

const BINARY_EXTENSIONS = [_][]const u8{
    ".exe", ".dll", ".so", ".dylib", ".o", ".a", ".zip", ".tar", ".gz", ".7z",
    ".rar", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".ico", ".pdf",
    ".mp3", ".opus", ".flac", ".wav", ".m4a", ".aac", ".mp4", ".m4v", ".avi",
    ".mov", ".mkv", ".webm", ".ttf", ".woff", ".woff2", ".db", ".sqlite",
};

const DEFAULT_IGNORES = [_][]const u8{
    ".git", ".zbed", ".bed", "node_modules", "vendor", "dist", "build", "target",
    ".cache", "__pycache__", ".venv", "model",
};

const FileType = enum {
    text,
    binary,
    ignored,
    too_large,
};

pub fn walkAndIndex(allocator: Allocator, root: []const u8, model: *const embed_mod.EmbedModel, index: *Index, options: WalkOptions, progress_fn: ?*const fn (usize) void) !void {
    var ignore = IgnoreFilter.init(allocator);
    defer ignore.deinit();

    for (DEFAULT_IGNORES) |pattern| {
        try ignore.add(pattern, false, true);
    }

    var gi_buf: [4096]u8 = undefined;
    const gi_path = std.fmt.bufPrint(&gi_buf, "{s}/.gitignore", .{root}) catch null;
    if (gi_path) |path| {
        ignore.loadFile(path) catch {};
    }

    var dir = try std.fs.cwd().openDir(root, .{ .iterate = true });
    defer dir.close();

    var scratch = embed_mod.EmbedScratch{};
    var quantized = embed_mod.QuantizedEmbedding{};
    try walkDir(allocator, dir, root, "", &ignore, model, index, options, &scratch, &quantized, progress_fn);
}

fn walkDir(
    allocator: Allocator,
    dir: std.fs.Dir,
    root: []const u8,
    rel_prefix: []const u8,
    ignore: *const IgnoreFilter,
    model: *const embed_mod.EmbedModel,
    index: *Index,
    options: WalkOptions,
    scratch: *embed_mod.EmbedScratch,
    quantized: *embed_mod.QuantizedEmbedding,
    progress_fn: ?*const fn (usize) void,
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
                try walkDir(allocator, subdir, root, rel_path, ignore, model, index, options, scratch, quantized, progress_fn);
            },
            .file => {
                if (ignore.shouldIgnore(rel_path, false)) continue;

                var full_buf: [4096]u8 = undefined;
                const full_path = try std.fmt.bufPrint(&full_buf, "{s}/{s}", .{ root, rel_path });
                const file_type = detectFileType(full_path, rel_path, options);
                switch (file_type) {
                    .ignored, .too_large => continue,
                    .binary => {
                        if (!options.search_binaries) continue;
                        try indexBinaryFile(allocator, rel_path, model, index, options, scratch, quantized);
                    },
                    .text => {
                        try indexTextFile(allocator, full_path, rel_path, model, index, options, scratch, quantized);
                    },
                }

                if (progress_fn) |callback| callback(index.count());
            },
            else => {},
        }
    }
}

fn indexBinaryFile(allocator: Allocator, rel_path: []const u8, model: *const embed_mod.EmbedModel, index: *Index, options: WalkOptions, scratch: *embed_mod.EmbedScratch, quantized: *embed_mod.QuantizedEmbedding) !void {
    const display_name = std.fs.path.basename(rel_path);
    const search_text = try normalizePathForSearch(allocator, rel_path);
    defer allocator.free(search_text);

    const valid = if (options.gpu_embedder) |gpu_embedder|
        try gpu_embedder.embedQuantized(model, search_text, scratch, quantized)
    else
        model.embedQuantizedWithScratch(search_text, scratch, quantized);
    if (valid == 0) return;

    try index.addDocumentQuantized(rel_path, 0, display_name, .binary, quantized.data[0..model.embed_dim], quantized.scale, quantized.norm);
}

fn indexTextFile(allocator: Allocator, full_path: []const u8, rel_path: []const u8, model: *const embed_mod.EmbedModel, index: *Index, options: WalkOptions, scratch: *embed_mod.EmbedScratch, quantized: *embed_mod.QuantizedEmbedding) !void {
    if (options.include_path_documents) {
        const display_name = std.fs.path.basename(rel_path);
        const search_text = try normalizePathForSearch(allocator, rel_path);
        defer allocator.free(search_text);

        const valid = if (options.gpu_embedder) |gpu_embedder|
            try gpu_embedder.embedQuantized(model, search_text, scratch, quantized)
        else
            model.embedQuantizedWithScratch(search_text, scratch, quantized);
        if (valid > 0) {
            try index.addDocumentQuantized(rel_path, 0, display_name, .path, quantized.data[0..model.embed_dim], quantized.scale, quantized.norm);
        }
    }

    const file = try std.fs.cwd().openFile(full_path, .{});
    defer file.close();

    const data = file.readToEndAlloc(allocator, options.max_file_size) catch return;
    defer allocator.free(data);

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

fn detectFileType(full_path: []const u8, rel_path: []const u8, options: WalkOptions) FileType {
    const stat = std.fs.cwd().statFile(full_path) catch return .ignored;
    if (@as(u64, @intCast(stat.size)) > options.max_file_size) return .too_large;

    if (isTextFile(rel_path)) return .text;
    if (isBinaryExtension(rel_path)) return .binary;
    return if (isLikelyBinary(full_path)) .binary else .text;
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
    return matchExtension(ext, &TEXT_EXTENSIONS);
}

fn isBinaryExtension(path: []const u8) bool {
    const ext = std.fs.path.extension(path);
    if (ext.len == 0) return false;
    return matchExtension(ext, &BINARY_EXTENSIONS);
}

fn matchExtension(ext: []const u8, extensions: []const []const u8) bool {
    var lower_buf: [32]u8 = undefined;
    const len = @min(ext.len, lower_buf.len);
    for (0..len) |i| lower_buf[i] = std.ascii.toLower(ext[i]);
    const lower = lower_buf[0..len];

    for (extensions) |candidate| {
        if (std.mem.eql(u8, lower, candidate)) return true;
    }
    return false;
}

fn isLikelyBinary(full_path: []const u8) bool {
    const file = std.fs.cwd().openFile(full_path, .{}) catch return true;
    defer file.close();

    var buf: [8192]u8 = undefined;
    const n = file.readAll(&buf) catch return true;
    if (n == 0) return false;

    if (std.mem.indexOfScalar(u8, buf[0..n], 0) != null) return true;

    var non_printable: usize = 0;
    for (buf[0..n]) |byte| {
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

    const IgnorePattern = struct {
        pattern: []const u8,
        negation: bool,
        dir_only: bool,
    };

    fn init(allocator: Allocator) IgnoreFilter {
        return .{
            .patterns = .{},
            .allocator = allocator,
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
        var ignored = false;
        for (self.patterns.items) |pattern| {
            if (pattern.dir_only and !is_dir) {
                if (!matchesDirectoryPattern(pattern.pattern, rel_path)) continue;
            } else if (!(matchGlob(pattern.pattern, rel_path) or matchBasename(pattern.pattern, rel_path))) {
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
