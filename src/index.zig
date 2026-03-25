const std = @import("std");
const Allocator = std.mem.Allocator;
const embed_mod = @import("embed.zig");
const search_mod = @import("search.zig");

/// Magic number for .zbed index files.
const INDEX_MAGIC: u32 = 0x5A424544; // "ZBED"
/// Current index format version.
const INDEX_VERSION: u8 = 1;

/// A single indexed line (document).
pub const Document = struct {
    /// File path (relative to indexed root)
    file_path: []const u8,
    /// 1-based line number
    line_num: u32,
    /// Line content
    content: []const u8,
};

/// Persistent semantic search index.
pub const Index = struct {
    /// All indexed documents.
    documents: std.ArrayList(Document),
    /// Flat f32 embeddings [n_docs * dim].
    embeddings: std.ArrayList(f32),
    /// Embedding dimension.
    dim: usize,
    /// Root path that was indexed.
    root_path: []const u8,
    /// Owned strings for documents.
    string_arena: std.heap.ArenaAllocator,

    allocator: Allocator,

    pub fn init(allocator: Allocator, dim: usize) Index {
        return .{
            .documents = std.ArrayList(Document).init(allocator),
            .embeddings = std.ArrayList(f32).init(allocator),
            .dim = dim,
            .root_path = "",
            .string_arena = std.heap.ArenaAllocator.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Index) void {
        self.documents.deinit();
        self.embeddings.deinit();
        self.string_arena.deinit();
    }

    /// Add a document with its embedding to the index.
    pub fn addDocument(self: *Index, file_path: []const u8, line_num: u32, content: []const u8, embedding: []const f32) !void {
        const arena = self.string_arena.allocator();
        const owned_path = try arena.dupe(u8, file_path);
        const owned_content = try arena.dupe(u8, content);

        try self.documents.append(.{
            .file_path = owned_path,
            .line_num = line_num,
            .content = owned_content,
        });
        try self.embeddings.appendSlice(embedding[0..self.dim]);
    }

    /// Number of indexed documents.
    pub fn count(self: *const Index) usize {
        return self.documents.items.len;
    }

    /// Build a FlatIndex for searching.
    pub fn buildSearchIndex(self: *const Index, allocator: Allocator) !search_mod.FlatIndex {
        var flat = search_mod.FlatIndex.init(allocator, self.dim);
        if (self.count() == 0) return flat;

        const data = try allocator.alloc(f32, self.embeddings.items.len);
        @memcpy(data, self.embeddings.items);
        try flat.build(data, self.count());
        return flat;
    }

    // ── Persistence ─────────────────────────────────────────────────

    /// Save index to .zbed/index.bin under the given directory.
    pub fn save(self: *const Index, dir_path: []const u8) !void {
        // Ensure .zbed directory exists
        var zbed_buf: [4096]u8 = undefined;
        const zbed_path = try std.fmt.bufPrint(&zbed_buf, "{s}/.zbed", .{dir_path});

        std.fs.cwd().makeDir(zbed_path) catch |err| {
            if (err != error.PathAlreadyExists) return err;
        };

        var path_buf: [4096]u8 = undefined;
        const full_path = try std.fmt.bufPrint(&path_buf, "{s}/.zbed/index.bin", .{dir_path});

        const file = try std.fs.cwd().createFile(full_path, .{});
        defer file.close();
        const writer = file.writer();

        // Header
        try writer.writeInt(u32, INDEX_MAGIC, .little);
        try writer.writeInt(u8, INDEX_VERSION, .little);
        try writer.writeInt(u32, @intCast(self.dim), .little);
        try writer.writeInt(u32, @intCast(self.count()), .little);

        // Documents
        for (self.documents.items) |doc| {
            try writeString(writer, doc.file_path);
            try writer.writeInt(u32, doc.line_num, .little);
            try writeString(writer, doc.content);
        }

        // Embeddings (raw f32 bytes)
        const emb_bytes = std.mem.sliceAsBytes(self.embeddings.items);
        try writer.writeAll(emb_bytes);
    }

    /// Load index from .zbed/index.bin under the given directory.
    pub fn load(self: *Index, dir_path: []const u8) !void {
        var path_buf: [4096]u8 = undefined;
        const full_path = try std.fmt.bufPrint(&path_buf, "{s}/.zbed/index.bin", .{dir_path});

        const file = try std.fs.cwd().openFile(full_path, .{});
        defer file.close();
        const reader = file.reader();

        // Header
        const magic = try reader.readInt(u32, .little);
        if (magic != INDEX_MAGIC) return error.InvalidMagic;

        const version = try reader.readInt(u8, .little);
        if (version != INDEX_VERSION) return error.UnsupportedVersion;

        const dim = try reader.readInt(u32, .little);
        const n_docs = try reader.readInt(u32, .little);

        self.dim = @intCast(dim);

        // Clear existing data
        self.documents.clearRetainingCapacity();
        self.embeddings.clearRetainingCapacity();

        const arena = self.string_arena.allocator();

        // Read documents
        for (0..n_docs) |_| {
            const file_path = try readString(reader, arena);
            const line_num = try reader.readInt(u32, .little);
            const content = try readString(reader, arena);

            try self.documents.append(.{
                .file_path = file_path,
                .line_num = line_num,
                .content = content,
            });
        }

        // Read embeddings
        const emb_count = n_docs * self.dim;
        try self.embeddings.resize(emb_count);
        const emb_bytes = std.mem.sliceAsBytes(self.embeddings.items);
        const bytes_read = try reader.readAll(emb_bytes);
        if (bytes_read != emb_bytes.len) return error.UnexpectedEof;
    }

    /// Check if a cached index exists for the given directory.
    pub fn exists(dir_path: []const u8) bool {
        var path_buf: [4096]u8 = undefined;
        const full_path = std.fmt.bufPrint(&path_buf, "{s}/.zbed/index.bin", .{dir_path}) catch return false;
        std.fs.cwd().access(full_path, .{}) catch return false;
        return true;
    }
};

// ── Wire helpers ─────────────────────────────────────────────────────

fn writeString(writer: anytype, s: []const u8) !void {
    try writer.writeInt(u32, @intCast(s.len), .little);
    try writer.writeAll(s);
}

fn readString(reader: anytype, allocator: Allocator) ![]const u8 {
    const len = try reader.readInt(u32, .little);
    if (len > 10 * 1024 * 1024) return error.StringTooLong;
    const buf = try allocator.alloc(u8, len);
    const n = try reader.readAll(buf);
    if (n != len) return error.UnexpectedEof;
    return buf;
}

// ── .gitignore-aware directory walking ──────────────────────────────

/// Known text file extensions for indexing.
const TEXT_EXTENSIONS = [_][]const u8{
    ".txt", ".md",  ".rst", ".tex",  ".go",   ".py",   ".js",   ".ts",
    ".jsx", ".tsx", ".c",   ".cpp",  ".h",    ".hpp",  ".rs",   ".rb",
    ".php", ".java",".cs",  ".swift",".kt",   ".scala",".zig",  ".nim",
    ".lua", ".r",   ".m",   ".hs",   ".ml",   ".elm",  ".clj",  ".ex",
    ".erl", ".fs",  ".vb",  ".dart", ".html", ".css",  ".scss", ".sass",
    ".less",".vue", ".xml", ".json", ".yaml", ".yml",  ".toml", ".ini",
    ".conf",".cfg", ".sh",  ".bash", ".zsh",  ".fish", ".ps1",  ".bat",
    ".cmd", ".sql", ".graphql",".proto",".cmake",".makefile",
};

/// Check if a file extension suggests a text file.
pub fn isTextFile(path: []const u8) bool {
    // Check known names without extension
    const basename = std.fs.path.basename(path);
    const known_names = [_][]const u8{
        "Makefile",  "Dockerfile", "Vagrantfile", "Rakefile", "Gemfile",
        "README",    "LICENSE",    "CHANGELOG",   "CONTRIBUTING",
        "Justfile",  "Taskfile",   "Procfile",    ".gitignore",
        ".gitattributes", ".editorconfig", ".eslintrc", ".prettierrc",
        "build.zig", "build.zig.zon",
    };
    for (known_names) |name| {
        if (std.mem.eql(u8, basename, name)) return true;
    }

    // Check extension
    const ext = std.fs.path.extension(path);
    if (ext.len == 0) return false;

    const ext_lower_buf = blk: {
        var buf: [32]u8 = undefined;
        const len = @min(ext.len, buf.len);
        for (0..len) |i| {
            buf[i] = std.ascii.toLower(ext[i]);
        }
        break :blk buf[0..len];
    };

    for (TEXT_EXTENSIONS) |te| {
        if (std.mem.eql(u8, ext_lower_buf, te)) return true;
    }
    return false;
}

/// Load .gitignore patterns from a file (simple glob matching).
pub const IgnoreFilter = struct {
    patterns: std.ArrayList(IgnorePattern),
    allocator: Allocator,

    const IgnorePattern = struct {
        pattern: []const u8,
        negation: bool,
        dir_only: bool,
    };

    pub fn init(allocator: Allocator) IgnoreFilter {
        return .{
            .patterns = std.ArrayList(IgnorePattern).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *IgnoreFilter) void {
        for (self.patterns.items) |p| {
            self.allocator.free(p.pattern);
        }
        self.patterns.deinit();
    }

    /// Load patterns from a .gitignore file.
    pub fn loadFile(self: *IgnoreFilter, path: []const u8) !void {
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

            // Strip leading slash (anchored pattern, treated same for simplicity)
            if (line.len > 0 and line[0] == '/') {
                line = line[1..];
            }

            if (line.len == 0) continue;
            const owned = try self.allocator.dupe(u8, line);
            try self.patterns.append(.{
                .pattern = owned,
                .negation = negation,
                .dir_only = dir_only,
            });
        }
    }

    /// Check if a relative path should be ignored.
    pub fn shouldIgnore(self: *const IgnoreFilter, rel_path: []const u8) bool {
        var ignored = false;
        for (self.patterns.items) |pat| {
            if (matchGlob(pat.pattern, rel_path) or matchBasename(pat.pattern, rel_path)) {
                ignored = !pat.negation;
            }
        }
        return ignored;
    }
};

/// Simple glob matching supporting * and **.
fn matchGlob(pattern: []const u8, path: []const u8) bool {
    // Simple cases
    if (std.mem.eql(u8, pattern, path)) return true;

    // Check if pattern matches the path basename
    if (std.mem.indexOf(u8, pattern, "/") == null) {
        // No slash in pattern - match against any path component
        return matchSimpleGlob(pattern, std.fs.path.basename(path));
    }

    // Pattern has slash - match against full relative path
    return matchSimpleGlob(pattern, path);
}

fn matchBasename(pattern: []const u8, path: []const u8) bool {
    const basename = std.fs.path.basename(path);
    return matchSimpleGlob(pattern, basename);
}

/// Match a simple glob pattern (with * wildcards) against a string.
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
            // Handle ** (match path separators too)
            if (pi + 1 < pattern.len and pattern[pi + 1] == '*') {
                pi += 2;
                // Skip optional / after **
                if (pi < pattern.len and pattern[pi] == '/') pi += 1;
                // ** matches everything including /
                star_p = pi;
                star_s = si;
            } else {
                star_p = pi + 1;
                star_s = si;
                pi += 1;
            }
        } else if (star_p) |sp| {
            pi = sp;
            star_s += 1;
            si = star_s;
        } else {
            return false;
        }
    }

    while (pi < pattern.len and pattern[pi] == '*') pi += 1;
    return pi == pattern.len;
}

/// Walk a directory tree, respecting .gitignore, and call `callback` for each text file line.
pub fn walkAndIndex(
    allocator: Allocator,
    root: []const u8,
    model: *const embed_mod.EmbedModel,
    index: *Index,
    progress_fn: ?*const fn (usize) void,
) !void {
    var ignore = IgnoreFilter.init(allocator);
    defer ignore.deinit();

    // Load .gitignore from root
    var gi_buf: [4096]u8 = undefined;
    const gi_path = std.fmt.bufPrint(&gi_buf, "{s}/.gitignore", .{root}) catch null;
    if (gi_path) |p| ignore.loadFile(p) catch {};

    // Always ignore .git and .zbed directories
    const git_pat = try allocator.dupe(u8, ".git");
    try ignore.patterns.append(.{ .pattern = git_pat, .negation = false, .dir_only = true });
    const zbed_pat = try allocator.dupe(u8, ".zbed");
    try ignore.patterns.append(.{ .pattern = zbed_pat, .negation = false, .dir_only = true });

    var dir = try std.fs.cwd().openDir(root, .{ .iterate = true });
    defer dir.close();

    try walkDir(allocator, dir, root, "", &ignore, model, index, progress_fn);
}

fn walkDir(
    allocator: Allocator,
    dir: std.fs.Dir,
    root: []const u8,
    rel_prefix: []const u8,
    ignore: *const IgnoreFilter,
    model: *const embed_mod.EmbedModel,
    index: *Index,
    progress_fn: ?*const fn (usize) void,
) !void {
    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        // Build relative path
        var rel_buf: [4096]u8 = undefined;
        const rel_path = if (rel_prefix.len > 0)
            try std.fmt.bufPrint(&rel_buf, "{s}/{s}", .{ rel_prefix, entry.name })
        else
            try std.fmt.bufPrint(&rel_buf, "{s}", .{entry.name});

        if (ignore.shouldIgnore(rel_path)) continue;

        switch (entry.kind) {
            .directory => {
                // Skip hidden directories
                if (entry.name.len > 0 and entry.name[0] == '.') continue;

                var subdir = dir.openDir(entry.name, .{ .iterate = true }) catch continue;
                defer subdir.close();
                try walkDir(allocator, subdir, root, rel_path, ignore, model, index, progress_fn);
            },
            .file => {
                if (!isTextFile(entry.name)) continue;

                // Build full path
                var full_buf: [4096]u8 = undefined;
                const full_path = if (root.len > 0)
                    try std.fmt.bufPrint(&full_buf, "{s}/{s}", .{ root, rel_path })
                else
                    rel_path;

                indexFile(allocator, full_path, rel_path, model, index) catch continue;

                if (progress_fn) |pf| pf(index.count());
            },
            else => {},
        }
    }
}

fn indexFile(
    allocator: Allocator,
    full_path: []const u8,
    rel_path: []const u8,
    model: *const embed_mod.EmbedModel,
    index: *Index,
) !void {
    const file = try std.fs.cwd().openFile(full_path, .{});
    defer file.close();

    // Read file (up to 10 MB)
    const data = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch return;
    defer allocator.free(data);

    var emb_buf: [embed_mod.MAX_EMBED_DIM]f32 = undefined;
    const dim = model.embed_dim;

    var lines = std.mem.splitScalar(u8, data, '\n');
    var line_num: u32 = 0;
    while (lines.next()) |line| {
        line_num += 1;
        // Skip very short or very long lines
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len < 3 or trimmed.len > 1200) continue;

        const valid = model.embed(trimmed, emb_buf[0..dim]);
        if (valid == 0) continue;

        try index.addDocument(rel_path, line_num, trimmed, emb_buf[0..dim]);
    }
}

// ─── tests ───────────────────────────────────────────────────────────
test "index save and load round-trip" {
    const allocator = std.testing.allocator;
    var idx = Index.init(allocator, 4);
    defer idx.deinit();

    const emb1 = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const emb2 = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    try idx.addDocument("test.txt", 1, "hello world", &emb1);
    try idx.addDocument("test.txt", 2, "foo bar", &emb2);

    // Save to temp dir
    const tmp_dir = "/tmp/zbed_test_idx";
    std.fs.cwd().makeDir(tmp_dir) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir) catch {};

    try idx.save(tmp_dir);

    // Load into new index
    var idx2 = Index.init(allocator, 4);
    defer idx2.deinit();
    try idx2.load(tmp_dir);

    try std.testing.expectEqual(@as(usize, 2), idx2.count());
    try std.testing.expectEqual(@as(usize, 4), idx2.dim);
    try std.testing.expectEqualStrings("hello world", idx2.documents.items[0].content);
    try std.testing.expectEqualStrings("foo bar", idx2.documents.items[1].content);
    try std.testing.expectEqual(@as(u32, 1), idx2.documents.items[0].line_num);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), idx2.embeddings.items[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), idx2.embeddings.items[7], 0.001);
}

test "gitignore pattern matching" {
    const allocator = std.testing.allocator;
    var filter = IgnoreFilter.init(allocator);
    defer filter.deinit();

    const p1 = try allocator.dupe(u8, "*.log");
    try filter.patterns.append(.{ .pattern = p1, .negation = false, .dir_only = false });
    const p2 = try allocator.dupe(u8, "node_modules");
    try filter.patterns.append(.{ .pattern = p2, .negation = false, .dir_only = true });
    const p3 = try allocator.dupe(u8, "build");
    try filter.patterns.append(.{ .pattern = p3, .negation = false, .dir_only = true });

    try std.testing.expect(filter.shouldIgnore("debug.log"));
    try std.testing.expect(filter.shouldIgnore("src/debug.log"));
    try std.testing.expect(filter.shouldIgnore("node_modules"));
    try std.testing.expect(!filter.shouldIgnore("src/main.go"));
}

test "isTextFile" {
    try std.testing.expect(isTextFile("main.go"));
    try std.testing.expect(isTextFile("src/lib.rs"));
    try std.testing.expect(isTextFile("Makefile"));
    try std.testing.expect(isTextFile("README"));
    try std.testing.expect(!isTextFile("image.png"));
    try std.testing.expect(!isTextFile("binary.exe"));
}
