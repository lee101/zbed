const std = @import("std");
const Allocator = std.mem.Allocator;
const tokenizer_mod = @import("tokenizer.zig");
const search_mod = @import("search.zig");

const Tokenizer = tokenizer_mod.Tokenizer;

pub const EMBED_DIM: usize = 512;
pub const MAX_EMBED_DIM: usize = 1024;
pub const MAX_TOKENS: usize = 256;

const SIMD_WIDTH = 8;
const VecF32 = @Vector(SIMD_WIDTH, f32);
const VecI8 = @Vector(SIMD_WIDTH, i8);
const VecI16 = @Vector(SIMD_WIDTH, i16);

pub const EmbedScratch = struct {
    token_buf: [MAX_TOKENS]i16 = undefined,
    float_buf: [MAX_EMBED_DIM]f32 = undefined,
};

pub const QuantizedEmbedding = struct {
    data: [MAX_EMBED_DIM]i8 = [_]i8{0} ** MAX_EMBED_DIM,
    scale: f32 = 1.0,
    norm: f32 = 0.0,
    valid_tokens: usize = 0,
};

pub const EmbedModel = struct {
    weights: []i8,
    scales: []f32,
    vocab_size: usize,
    embed_dim: usize,
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

    pub fn loadFromDir(self: *EmbedModel, model_dir: []const u8) !void {
        const st_names = [_][]const u8{
            "modelint8_512dim.safetensors",
            "modelint8_256dim.safetensors",
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

        var tok_path_buf: [1024]u8 = undefined;
        const tok_path = try std.fmt.bufPrint(&tok_path_buf, "{s}/tokenizer.json", .{model_dir});
        try self.tokenizer.loadFromFile(tok_path);
    }

    pub fn loadSafetensors(self: *EmbedModel, path: []const u8) !void {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        var hdr_len_buf: [8]u8 = undefined;
        const hl_read = try file.readAll(&hdr_len_buf);
        if (hl_read != 8) return error.UnexpectedEof;
        const header_len = std.mem.littleToNative(u64, std.mem.bytesToValue(u64, &hdr_len_buf));
        if (header_len > 16 * 1024 * 1024) return error.HeaderTooLarge;

        const header_bytes = try self.allocator.alloc(u8, @intCast(header_len));
        defer self.allocator.free(header_bytes);
        const bytes_read = try file.readAll(header_bytes);
        if (bytes_read != @as(usize, @intCast(header_len))) return error.UnexpectedEof;

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

        var v_size: usize = undefined;
        var e_dim: usize = undefined;
        if (emb_shape[0] > 0 and emb_shape[1] > 0) {
            v_size = @intCast(emb_shape[0]);
            e_dim = @intCast(emb_shape[1]);
        } else {
            v_size = @intCast(scale_size / 4);
            if (v_size == 0) return error.InvalidModel;
            e_dim = @intCast(emb_size / v_size);
        }

        if (e_dim > MAX_EMBED_DIM) return error.DimensionTooLarge;

        self.weights = try self.allocator.alloc(i8, @intCast(emb_size));
        try file.seekTo(data_base + emb_start);
        const w_read = try file.readAll(std.mem.sliceAsBytes(self.weights));
        if (w_read != emb_size) return error.UnexpectedEof;

        self.scales = try self.allocator.alloc(f32, v_size);
        try file.seekTo(data_base + scale_start);
        const s_bytes = std.mem.sliceAsBytes(self.scales);
        const s_read = try file.readAll(s_bytes);
        if (s_read != scale_size) return error.UnexpectedEof;

        self.vocab_size = v_size;
        self.embed_dim = e_dim;
    }

    pub fn embed(self: *const EmbedModel, text: []const u8, result: []f32) usize {
        var scratch = EmbedScratch{};
        return self.embedWithScratch(text, &scratch, result);
    }

    pub fn embedWithScratch(self: *const EmbedModel, text: []const u8, scratch: *EmbedScratch, result: []f32) usize {
        const n_tokens = self.tokenizer.tokenize(text, scratch.token_buf[0..]);
        return self.embedTokens(scratch.token_buf[0..n_tokens], result);
    }

    pub fn embedQuantized(self: *const EmbedModel, text: []const u8, out: *QuantizedEmbedding) usize {
        var scratch = EmbedScratch{};
        return self.embedQuantizedWithScratch(text, &scratch, out);
    }

    pub fn embedQuantizedWithScratch(self: *const EmbedModel, text: []const u8, scratch: *EmbedScratch, out: *QuantizedEmbedding) usize {
        const valid = self.embedWithScratch(text, scratch, scratch.float_buf[0..self.embed_dim]);
        out.scale = quantize(scratch.float_buf[0..self.embed_dim], out.data[0..self.embed_dim]);
        out.norm = search_mod.quantizedNorm(out.data[0..self.embed_dim]);
        out.valid_tokens = valid;
        return valid;
    }

    pub fn embedTokens(self: *const EmbedModel, tokens: []const i16, result: []f32) usize {
        const dim = self.embed_dim;
        const out_dim = @min(dim, result.len);
        @memset(result[0..out_dim], 0);

        var valid: usize = 0;
        for (tokens) |tok| {
            if (tok < 0 or @as(usize, @intCast(tok)) >= self.vocab_size) continue;

            const tid: usize = @intCast(tok);
            const scale = self.scales[tid];
            const row = self.weights[tid * dim .. (tid + 1) * dim];

            var j: usize = 0;
            while (j + SIMD_WIDTH <= out_dim) : (j += SIMD_WIDTH) {
                const raw: VecI8 = row[j..][0..SIMD_WIDTH].*;
                const wide: VecI16 = raw;
                const floats: VecF32 = @floatFromInt(wide);
                const scale_v: VecF32 = @splat(scale);
                const cur: VecF32 = result[j..][0..SIMD_WIDTH].*;
                result[j..][0..SIMD_WIDTH].* = cur + floats * scale_v;
            }
            while (j < out_dim) : (j += 1) {
                result[j] += @as(f32, @floatFromInt(row[j])) * scale;
            }
            valid += 1;
        }

        if (valid > 1) {
            const inv: f32 = 1.0 / @as(f32, @floatFromInt(valid));
            var j: usize = 0;
            while (j + SIMD_WIDTH <= out_dim) : (j += SIMD_WIDTH) {
                const v: VecF32 = result[j..][0..SIMD_WIDTH].*;
                const inv_v: VecF32 = @splat(inv);
                result[j..][0..SIMD_WIDTH].* = v * inv_v;
            }
            while (j < out_dim) : (j += 1) {
                result[j] *= inv;
            }
        }

        return valid;
    }
};

pub fn quantize(vec: []const f32, out: []i8) f32 {
    const dim = @min(vec.len, out.len);
    if (dim == 0) return 1.0;

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

test "quantize round-trip" {
    const vec = [_]f32{ 1.0, -0.5, 0.25, 0.0, -1.0, 0.75, -0.125, 0.0 };
    var q: [8]i8 = undefined;
    const scale = quantize(&vec, &q);

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

    var result: [4]f32 = undefined;
    const n = model.embedTokens(&.{}, &result);
    try std.testing.expectEqual(@as(usize, 0), n);

    const tokens = [_]i16{0};
    const n2 = model.embedTokens(&tokens, &result);
    try std.testing.expectEqual(@as(usize, 1), n2);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result[1], 0.01);
}

test "embed quantized with scratch" {
    const allocator = std.testing.allocator;
    var model = EmbedModel.init(allocator);
    defer model.deinit();

    const dim = 8;
    const vocab = 3;
    model.weights = try allocator.alloc(i8, vocab * dim);
    model.scales = try allocator.alloc(f32, vocab);
    model.vocab_size = vocab;
    model.embed_dim = dim;

    for (0..dim) |i| model.weights[0 * dim + i] = 10;
    for (0..dim) |i| model.weights[1 * dim + i] = 20;
    for (0..dim) |i| model.weights[2 * dim + i] = 30;
    model.scales[0] = 0.1;
    model.scales[1] = 0.1;
    model.scales[2] = 0.1;

    const tokens = [_]i16{ 0, 1, 2 };
    var scratch = EmbedScratch{};
    const n = model.embedTokens(&tokens, scratch.float_buf[0..dim]);
    try std.testing.expectEqual(@as(usize, 3), n);

    var out = QuantizedEmbedding{};
    out.scale = quantize(scratch.float_buf[0..dim], out.data[0..dim]);
    out.norm = search_mod.quantizedNorm(out.data[0..dim]);
    out.valid_tokens = n;
    try std.testing.expect(out.scale > 0);
    try std.testing.expect(out.norm > 0);
}
