const std = @import("std");
const Allocator = std.mem.Allocator;
const embed_mod = @import("embed.zig");
const search_mod = @import("search.zig");
const index_mod = @import("index.zig");

pub const ServerState = struct {
    model: *embed_mod.EmbedModel,
    idx: *index_mod.Index,
    allocator: Allocator,

    pub fn init(allocator: Allocator, model: *embed_mod.EmbedModel, idx: *index_mod.Index) ServerState {
        return .{ .model = model, .idx = idx, .allocator = allocator };
    }
};

pub fn run(allocator: Allocator, state: *ServerState, port: u16) !void {
    const addr = std.net.Address.initIp4(.{ 0, 0, 0, 0 }, port);
    var server = try addr.listen(.{ .reuse_address = true });
    defer server.deinit();

    const w = std.fs.File.stderr().deprecatedWriter();
    try w.print("zbed server on :{d}\n", .{port});
    try w.print("  POST /index  POST /search  POST /similarity  GET /status\n", .{});

    while (true) {
        const conn = server.accept() catch continue;
        handleConn(allocator, state, conn) catch {};
        conn.stream.close();
    }
}

fn handleConn(allocator: Allocator, state: *ServerState, conn: std.net.Server.Connection) !void {
    var read_buf: [16384]u8 = undefined;
    var write_buf: [16384]u8 = undefined;
    var stream_reader = std.net.Stream.Reader.init(conn.stream, &read_buf);
    var stream_writer = std.net.Stream.Writer.init(conn.stream, &write_buf);

    var srv = std.http.Server.init(stream_reader.interface(), &stream_writer.interface);
    var req = srv.receiveHead() catch return;
    const target = req.head.target;

    if (std.mem.eql(u8, target, "/index") and req.head.method == .POST) {
        try handleIndex(allocator, state, &req);
    } else if (std.mem.eql(u8, target, "/search") and req.head.method == .POST) {
        try handleSearch(allocator, state, &req);
    } else if (std.mem.eql(u8, target, "/similarity") and req.head.method == .POST) {
        try handleSimilarity(allocator, state, &req);
    } else if (std.mem.eql(u8, target, "/status")) {
        try handleStatus(state, &req);
    } else {
        try req.respond("{\"error\":\"not found\"}", .{
            .status = .not_found,
            .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }},
        });
    }
}

const JSON_HDR = [_]std.http.Header{.{ .name = "content-type", .value = "application/json" }};

fn readBody(allocator: Allocator, req: *std.http.Server.Request) ![]u8 {
    var body_buf: [8192]u8 = undefined;
    const body_reader = req.readerExpectNone(&body_buf);
    return body_reader.allocRemaining(allocator, std.Io.Limit.limited(64 * 1024 * 1024)) catch return allocator.alloc(u8, 0);
}

fn respondJson(req: *std.http.Server.Request, body: []const u8) !void {
    try req.respond(body, .{ .extra_headers = &JSON_HDR });
}

fn respondErr(req: *std.http.Server.Request, status: std.http.Status, msg: []const u8) !void {
    try req.respond(msg, .{ .status = status, .extra_headers = &JSON_HDR });
}

fn handleIndex(allocator: Allocator, state: *ServerState, req: *std.http.Server.Request) !void {
    const body = readBody(allocator, req) catch {
        return respondErr(req, .bad_request, "{\"error\":\"read failed\"}");
    };
    defer if (body.len > 0) allocator.free(body);

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{
        .allocate = .alloc_always, .max_value_len = null,
    }) catch return respondErr(req, .bad_request, "{\"error\":\"invalid json\"}");
    defer parsed.deinit();

    const docs_val = parsed.value.object.get("documents") orelse
        return respondErr(req, .bad_request, "{\"error\":\"missing documents\"}");

    var indexed: usize = 0;
    var timer = try std.time.Timer.start();

    for (docs_val.array.items) |doc_val| {
        const doc_obj = doc_val.object;
        const text = if (doc_obj.get("text")) |t| switch (t) {
            .string => |s| s,
            else => continue,
        } else continue;

        const id: u32 = if (doc_obj.get("id")) |iv| switch (iv) {
            .integer => |v| @intCast(@as(u32, @truncate(@as(u64, @bitCast(v))))),
            else => @intCast(state.idx.count()),
        } else @intCast(state.idx.count());

        var qe = embed_mod.QuantizedEmbedding{};
        if (state.model.embedQuantized(text, &qe) == 0) continue;

        state.idx.addDocumentQuantized("api", id, text, .text, qe.data[0..state.model.embed_dim], qe.scale, qe.norm) catch continue;
        indexed += 1;
    }

    var buf: [256]u8 = undefined;
    const resp = std.fmt.bufPrint(&buf, "{{\"indexed\":{d},\"latency_us\":{d}}}", .{ indexed, timer.read() / 1000 }) catch "{\"indexed\":0}";
    try respondJson(req, resp);
}

fn handleSearch(allocator: Allocator, state: *ServerState, req: *std.http.Server.Request) !void {
    const body = readBody(allocator, req) catch {
        return respondErr(req, .bad_request, "{\"error\":\"read failed\"}");
    };
    defer if (body.len > 0) allocator.free(body);

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{
        .allocate = .alloc_always, .max_value_len = null,
    }) catch return respondErr(req, .bad_request, "{\"error\":\"invalid json\"}");
    defer parsed.deinit();

    const query_str = if (parsed.value.object.get("query")) |q| switch (q) {
        .string => |s| s,
        else => "",
    } else "";
    if (query_str.len == 0) return respondErr(req, .bad_request, "{\"error\":\"missing query\"}");

    const k: usize = if (parsed.value.object.get("k")) |kv| switch (kv) {
        .integer => |v| @intCast(@max(1, @min(100, v))),
        else => 10,
    } else 10;

    if (state.idx.count() == 0) return respondJson(req, "{\"results\":[],\"latency_us\":0}");

    var timer = try std.time.Timer.start();
    var qe = embed_mod.QuantizedEmbedding{};
    if (state.model.embedQuantized(query_str, &qe) == 0) return respondJson(req, "{\"results\":[],\"latency_us\":0}");

    const search_index = state.idx.buildSearchIndex();
    var results: [100]search_mod.SearchResult = undefined;
    const n = search_index.search(qe.data[0..state.model.embed_dim], qe.norm, @min(k, 100), 0.0, &results);
    const elapsed_us = timer.read() / 1000;

    var resp: std.ArrayListUnmanaged(u8) = .{};
    defer resp.deinit(allocator);
    const w = resp.writer(allocator);

    try w.print("{{\"results\":[", .{});
    for (results[0..n], 0..) |r, i| {
        if (i > 0) try w.writeByte(',');
        const doc = state.idx.documents.items[r.doc_idx];
        try w.print("{{\"id\":{d},\"text\":", .{doc.line_num});
        try writeJsonStr(w, doc.content);
        try w.print(",\"similarity\":{d:.6}}}", .{r.score});
    }
    try w.print("],\"latency_us\":{d}}}", .{elapsed_us});
    try respondJson(req, resp.items);
}

fn handleSimilarity(allocator: Allocator, state: *ServerState, req: *std.http.Server.Request) !void {
    const body = readBody(allocator, req) catch {
        return respondErr(req, .bad_request, "{\"error\":\"read failed\"}");
    };
    defer if (body.len > 0) allocator.free(body);

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{
        .allocate = .alloc_always, .max_value_len = null,
    }) catch return respondErr(req, .bad_request, "{\"error\":\"invalid json\"}");
    defer parsed.deinit();

    const text1 = if (parsed.value.object.get("text1")) |t| switch (t) { .string => |s| s, else => "" } else "";
    const text2 = if (parsed.value.object.get("text2")) |t| switch (t) { .string => |s| s, else => "" } else "";
    if (text1.len == 0 or text2.len == 0) return respondErr(req, .bad_request, "{\"error\":\"missing text1/text2\"}");

    const dim = state.model.embed_dim;
    var emb1: [embed_mod.MAX_EMBED_DIM]f32 = undefined;
    var emb2: [embed_mod.MAX_EMBED_DIM]f32 = undefined;
    _ = state.model.embed(text1, emb1[0..dim]);
    _ = state.model.embed(text2, emb2[0..dim]);
    const sim = search_mod.cosineSimilarity(emb1[0..dim], emb2[0..dim]);

    var buf: [128]u8 = undefined;
    const resp = std.fmt.bufPrint(&buf, "{{\"similarity\":{d:.6}}}", .{sim}) catch "{\"similarity\":0}";
    try respondJson(req, resp);
}

fn handleStatus(state: *ServerState, req: *std.http.Server.Request) !void {
    var buf: [512]u8 = undefined;
    const resp = std.fmt.bufPrint(&buf, "{{\"documents\":{d},\"dimensions\":{d}}}", .{
        state.idx.count(), state.idx.dim,
    }) catch "{\"error\":\"format\"}";
    try respondJson(req, resp);
}

fn writeJsonStr(writer: anytype, s: []const u8) !void {
    try writer.writeByte('"');
    for (s) |c| {
        switch (c) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => if (c < 0x20) {
                try writer.print("\\u{x:0>4}", .{c});
            } else {
                try writer.writeByte(c);
            },
        }
    }
    try writer.writeByte('"');
}
