const std = @import("std");
const Allocator = std.mem.Allocator;
const tokenizer_mod = @import("tokenizer.zig");
const Tokenizer = tokenizer_mod.Tokenizer;

/// Embedding dimension (matches static-retrieval-mrl-en-v1 int8 model).
pub const EMBED_DIM: usize = 256;
/// Maximum supported embedding dimension for the model file.
pub const MAX_EMBED_DIM: usize = 1024;

/// SIMD vector width for f32 operations.
const SIMD_WIDTH = 8;
const VecF32 = @Vector(SIMD_WIDTH, f32);
const VecI8 = @Vector(SIMD_WIDTH, i8);
const VecI16 = @Vector(SIMD_WIDTH, i16);

/// Int8 embedding model loaded from safetensors format.
/// Stores per-token int8 embeddings with per-token float32 scales.
pub const EmbedModel = struct {
    /// Flat int8 weights: [vocab_size * embed_dim]
    weights: []i8,
    /// Per-token quantization scales: [vocab_size]
    scales: []f32,
    /// Vocabulary size
    vocab_size: usize,
    /// Actual embedding dimension from model file
    embed_dim: usize,
    /// Tokenizer
    tokenizer: Tokenizer,

    allocator: Allocator,

    pub fn init(allocator: Allocator) EmbedModel {
        return .{
            .weights = &.{},
            .scales = &.{},
            .vocab_size = 0,
            .embed_dim = 0,
            .tokenizer = Tokenizer.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *EmbedModel) void {
        if (self.weights.len > 0) self.allocator.free(self.weights);
        if (self.scales.len > 0) self.allocator.free(self.scales);
        self.tokenizer.deinit();
    }

    /// Load model from model directory containing:
    ///   - modelint8_512dim.safetensors (or similar)
    ///   - tokenizer.json
    pub fn loadFromDir(self: *EmbedModel, model_dir: []const u8) !void {
        // Try several safetensors filenames
        const st_names = [_][]const u8{
            "modelint8_256dim.safetensors",
            "modelint8_512dim.safetensors",
            "model.safetensors",
        };

        var st_path_buf: [1024]u8 = undefined;
        var loaded = false;

        for (st_names) |name| {
            const st_path = std.fmt.bufPrint(&st_path_buf, "{s}/{s}", .{ model_dir, name }) catch continue;
            self.loadSafetensors(st_path) catch continue;
            loaded = true;
            break;
        }

        if (!loaded) return error.ModelNotFound;

        // Load tokenizer
        var tok_path_buf: [1024]u8 = undefined;
        const tok_path = try std.fmt.bufPrint(&tok_path_buf, "{s}/tokenizer.json", .{model_dir});
        try self.tokenizer.loadFromFile(tok_path);
    }

    /// Load int8 embeddings from safetensors file.
    /// Format:
    ///   [8 bytes: header_length (little-endian u64)]
    ///   [header_length bytes: JSON metadata]
    ///   [tensor data]
    pub fn loadSafetensors(self: *EmbedModel, path: []const u8) !void {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        const reader = file.reader();

        // Read header length
        const header_len = try reader.readInt(u64, .little);
        if (header_len > 16 * 1024 * 1024) return error.HeaderTooLarge;

        // Read header JSON
        const header_bytes = try self.allocator.alloc(u8, @intCast(header_len));
        defer self.allocator.free(header_bytes);
        const bytes_read = try reader.readAll(header_bytes);
        if (bytes_read != @as(usize, @intCast(header_len))) return error.UnexpectedEof;

        // Parse header to find tensor metadata
        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, header_bytes, .{
            .allocate = .alloc_always,
            .max_value_len = null,
        });
        defer parsed.deinit();

        const root = parsed.value.object;

        var emb_start: u64 = 0;
        var emb_end: u64 = 0;
        var scale_start: u64 = 0;
        var scale_end: u64 = 0;
        var emb_shape: [2]u64 = .{ 0, 0 };
        var found_emb = false;
        var found_scale = false;

        var it = root.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (std.mem.eql(u8, name, "__metadata__")) continue;

            const info = entry.value_ptr.object;

            const offsets = info.get("data_offsets") orelse continue;
            const offsets_arr = offsets.array.items;
            if (offsets_arr.len < 2) continue;

            const start: u64 = @intCast(offsets_arr[0].integer);
            const end: u64 = @intCast(offsets_arr[1].integer);

            if (std.mem.indexOf(u8, name, "weight") != null or
                (std.mem.indexOf(u8, name, "embedding") != null and std.mem.indexOf(u8, name, "scale") == null))
            {
                emb_start = start;
                emb_end = end;
                if (info.get("shape")) |shape| {
                    const shape_arr = shape.array.items;
                    if (shape_arr.len >= 2) {
                        emb_shape[0] = @intCast(shape_arr[0].integer);
                        emb_shape[1] = @intCast(shape_arr[1].integer);
                    }
                }
                found_emb = true;
            } else if (std.mem.indexOf(u8, name, "scale") != null) {
                scale_start = start;
                scale_end = end;
                found_scale = true;
            }
        }

        if (!found_emb or !found_scale) return error.TensorNotFound;

        const data_base: u64 = 8 + header_len;
        const emb_size = emb_end - emb_start;
        const scale_size = scale_end - scale_start;

        // Determine dimensions from shape or data
        var v_size: usize = undefined;
        var e_dim: usize = undefined;
        if (emb_shape[0] > 0 and emb_shape[1] > 0) {
            v_size = @intCast(emb_shape[0]);
            e_dim = @intCast(emb_shape[1]);
        } else {
            // Infer from scales size (4 bytes per f32)
            v_size = @intCast(scale_size / 4);
            if (v_size == 0) return error.InvalidModel;
            e_dim = @intCast(emb_size / v_size);
        }

        if (e_dim > MAX_EMBED_DIM) return error.DimensionTooLarge;

        // Read embeddings
        self.weights = try self.allocator.alloc(i8, @intCast(emb_size));
        try file.seekTo(data_base + emb_start);
        const w_read = try file.reader().readAll(std.mem.sliceAsBytes(self.weights));
        if (w_read != emb_size) return error.UnexpectedEof;

        // Read scales
        self.scales = try self.allocator.alloc(f32, v_size);
        try file.seekTo(data_base + scale_start);
        const s_bytes = std.mem.sliceAsBytes(self.scales);
        const s_read = try file.reader().readAll(s_bytes);
        if (s_read != scale_size) return error.UnexpectedEof;

        self.vocab_size = v_size;
        self.embed_dim = e_dim;
    }

    /// Embed text into a float32 vector using mean pooling.
    /// Returns the number of valid tokens processed.
    pub fn embed(self: *const EmbedModel, text: []const u8, result: []f32) usize {
        var token_buf: [512]i16 = undefined;
        const n_tokens = self.tokenizer.tokenize(text, &token_buf);
        return self.embedTokens(token_buf[0..n_tokens], result);
    }

    /// Embed pre-tokenized IDs into result buffer using SIMD-accelerated
    /// dequantization and mean pooling.
    pub fn embedTokens(self: *const EmbedModel, tokens: []const i16, result: []f32) usize {
        const dim = self.embed_dim;
        const out_dim = @min(dim, result.len);

        // Zero the result
        @memset(result[0..out_dim], 0);

        var valid: usize = 0;
        for (tokens) |tok| {
            if (tok < 0 or @as(usize, @intCast(tok)) >= self.vocab_size) continue;
            const tid: usize = @intCast(tok);
            const scale = self.scales[tid];
            const row = self.weights[tid * dim .. (tid + 1) * dim];

            // SIMD accumulation: i8 → i16 (widen) → f32 → scale+accumulate
            var j: usize = 0;
            while (j + SIMD_WIDTH <= out_dim) : (j += SIMD_WIDTH) {
                const raw: VecI8 = row[j..][0..SIMD_WIDTH].*;
                const wide: VecI16 = raw; // widening coercion i8 → i16
                const floats: VecF32 = @floatFromInt(wide);
                const scale_v: VecF32 = @splat(scale);
                const cur: VecF32 = result[j..][0..SIMD_WIDTH].*;
                result[j..][0..SIMD_WIDTH].* = cur + floats * scale_v;
            }
            // Scalar remainder
            while (j < out_dim) : (j += 1) {
                result[j] += @as(f32, @floatFromInt(row[j])) * scale;
            }
            valid += 1;
        }

        // Mean pooling
        if (valid > 1) {
            const inv: f32 = 1.0 / @as(f32, @floatFromInt(valid));
            var j: usize = 0;
            while (j + SIMD_WIDTH <= out_dim) : (j += SIMD_WIDTH) {
                const v: VecF32 = result[j..][0..SIMD_WIDTH].*;
                const s: VecF32 = @splat(inv);
                result[j..][0..SIMD_WIDTH].* = v * s;
            }
            while (j < out_dim) : (j += 1) {
                result[j] *= inv;
            }
        }

        return valid;
    }

    /// Quantize a float32 embedding back to int8 + scale.
    pub fn quantize(vec: []const f32, out: []i8) f32 {
        const dim = @min(vec.len, out.len);
        if (dim == 0) return 1.0;

        // Find max absolute value
        var max_abs: f32 = 0;
        for (vec[0..dim]) |v| {
            const a = @abs(v);
            if (a > max_abs) max_abs = a;
        }

        if (max_abs == 0) {
            @memset(out[0..dim], 0);
            return 1.0;
        }

        const scale = max_abs / 127.0;
        const inv_scale = 127.0 / max_abs;

        for (0..dim) |i| {
            const scaled = vec[i] * inv_scale;
            const clamped = @max(-127.0, @min(127.0, scaled));
            out[i] = @intFromFloat(clamped);
        }

        return scale;
    }
};

// ─── tests ───────────────────────────────────────────────────────────
test "quantize round-trip" {
    var vec = [_]f32{ 1.0, -0.5, 0.25, 0.0, -1.0, 0.75, -0.125, 0.0 };
    var q: [8]i8 = undefined;
    const scale = EmbedModel.quantize(&vec, &q);

    // Dequantize and check approximate match
    for (0..8) |i| {
        const reconstructed = @as(f32, @floatFromInt(q[i])) * scale;
        const diff = @abs(reconstructed - vec[i]);
        try std.testing.expect(diff < 0.02);
    }
}

test "embed tokens zero input" {
    const allocator = std.testing.allocator;
    var model = EmbedModel.init(allocator);
    defer model.deinit();

    // Create tiny model: vocab=2, dim=4
    model.weights = try allocator.alloc(i8, 8);
    @memset(model.weights, 0);
    model.weights[0] = 10;
    model.weights[1] = 20;
    model.weights[2] = 30;
    model.weights[3] = 40;
    model.scales = try allocator.alloc(f32, 2);
    model.scales[0] = 0.1;
    model.scales[1] = 0.2;
    model.vocab_size = 2;
    model.embed_dim = 4;

    // Empty tokens
    var result: [4]f32 = undefined;
    const n = model.embedTokens(&.{}, &result);
    try std.testing.expectEqual(@as(usize, 0), n);

    // Single token
    const tokens = [_]i16{0};
    const n2 = model.embedTokens(&tokens, &result);
    try std.testing.expectEqual(@as(usize, 1), n2);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result[1], 0.01);
}

test "embed tokens SIMD path (dim=16)" {
    const allocator = std.testing.allocator;
    var model = EmbedModel.init(allocator);
    defer model.deinit();

    const dim = 16;
    const vocab = 3;

    model.weights = try allocator.alloc(i8, vocab * dim);
    model.scales = try allocator.alloc(f32, vocab);
    model.vocab_size = vocab;
    model.embed_dim = dim;

    // Token 0: all 10s, scale=0.1 → each element = 1.0
    for (0..dim) |i| model.weights[0 * dim + i] = 10;
    model.scales[0] = 0.1;

    // Token 1: all 20s, scale=0.05 → each element = 1.0
    for (0..dim) |i| model.weights[1 * dim + i] = 20;
    model.scales[1] = 0.05;

    // Token 2: values 1..16, scale=1.0
    for (0..dim) |i| model.weights[2 * dim + i] = @intCast(i + 1);
    model.scales[2] = 1.0;

    // Single token: should get exact values via SIMD path
    var result: [dim]f32 = undefined;
    const tokens_single = [_]i16{2};
    const n1 = model.embedTokens(&tokens_single, &result);
    try std.testing.expectEqual(@as(usize, 1), n1);
    for (0..dim) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(i + 1)), result[i], 0.001);
    }

    // Two tokens (0 and 1): both contribute 1.0 per element, mean = 1.0
    const tokens_pair = [_]i16{ 0, 1 };
    const n2 = model.embedTokens(&tokens_pair, &result);
    try std.testing.expectEqual(@as(usize, 2), n2);
    for (0..dim) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[i], 0.001);
    }

    // Negative int8 values: Token 0 with negative weights
    for (0..dim) |i| model.weights[0 * dim + i] = -50;
    model.scales[0] = 0.02; // -50 * 0.02 = -1.0
    const tokens_neg = [_]i16{0};
    const n3 = model.embedTokens(&tokens_neg, &result);
    try std.testing.expectEqual(@as(usize, 1), n3);
    for (0..dim) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, -1.0), result[i], 0.001);
    }
}

test "embed tokens mean pooling with three tokens" {
    const allocator = std.testing.allocator;
    var model = EmbedModel.init(allocator);
    defer model.deinit();

    const dim = 16;
    const vocab = 3;

    model.weights = try allocator.alloc(i8, vocab * dim);
    model.scales = try allocator.alloc(f32, vocab);
    model.vocab_size = vocab;
    model.embed_dim = dim;

    // Token 0: all 30s, scale=0.1 → 3.0
    // Token 1: all 60s, scale=0.1 → 6.0
    // Token 2: all 90s, scale=0.1 → 9.0
    // Mean = (3 + 6 + 9) / 3 = 6.0
    for (0..dim) |i| model.weights[0 * dim + i] = 30;
    for (0..dim) |i| model.weights[1 * dim + i] = 60;
    for (0..dim) |i| model.weights[2 * dim + i] = 90;
    model.scales[0] = 0.1;
    model.scales[1] = 0.1;
    model.scales[2] = 0.1;

    var result: [dim]f32 = undefined;
    const tokens = [_]i16{ 0, 1, 2 };
    const n = model.embedTokens(&tokens, &result);
    try std.testing.expectEqual(@as(usize, 3), n);
    for (0..dim) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 6.0), result[i], 0.001);
    }
}
