const std = @import("std");
const Allocator = std.mem.Allocator;
const embed_mod = @import("embed.zig");
const search_mod = @import("search.zig");
const index_mod = @import("index.zig");
const watcher_mod = @import("watcher.zig");

pub const ServerState = struct {
    model: *embed_mod.EmbedModel,
    idx: *index_mod.Index,
    allocator: Allocator,
    root_path: []const u8,
    walk_options: index_mod.WalkOptions,
    watcher: ?watcher_mod.Watcher,
    files_changed: usize,
    dirty: bool,
    last_save_ns: u64,
    save_interval_ns: u64,

    pub fn init(allocator: Allocator, model: *embed_mod.EmbedModel, idx: *index_mod.Index, root_path: []const u8, walk_options: index_mod.WalkOptions) ServerState {
        return .{
            .model = model,
            .idx = idx,
            .allocator = allocator,
            .root_path = root_path,
            .walk_options = walk_options,
            .watcher = null,
            .files_changed = 0,
            .dirty = false,
            .last_save_ns = 0,
            .save_interval_ns = 30 * std.time.ns_per_s,
        };
    }

    pub fn maybeAutoSave(self: *ServerState) void {
        if (!self.dirty or self.root_path.len == 0) return;
        const now: u64 = @intCast(std.time.nanoTimestamp());
        if (now - self.last_save_ns < self.save_interval_ns) return;
        self.idx.save(self.root_path) catch |err| {
            log("auto-save failed: {}\n", .{err});
            return;
        };
        self.last_save_ns = now;
        self.dirty = false;
        log("auto-saved index ({d} docs)\n", .{self.idx.count()});
    }

    pub fn startWatching(self: *ServerState) !void {
        self.watcher = try watcher_mod.Watcher.init(self.allocator, self.root_path);
        log("watching {s} for changes\n", .{self.root_path});
    }

    pub fn processFileEvents(self: *ServerState) usize {
        var w = self.watcher orelse return 0;
        var events: [64]watcher_mod.Event = undefined;
        const n = w.poll(&events);
        if (n == 0) return 0;

        // Batch deletes into a single compaction pass
        var deletes: [64][]const u8 = undefined;
        var n_deletes: usize = 0;
        var updated: usize = 0;

        for (events[0..n]) |ev| {
            switch (ev.kind) {
                .deleted => {
                    if (n_deletes < deletes.len) {
                        deletes[n_deletes] = ev.rel_path;
                        n_deletes += 1;
                    }
                },
                .created, .modified => {
                    const added = self.idx.reindexFile(
                        self.allocator,
                        self.root_path,
                        ev.rel_path,
                        self.model,
                        self.walk_options,
                    ) catch 0;
                    log("~ {s} ({d} docs)\n", .{ ev.rel_path, added });
                    updated += added;
                },
            }
        }

        if (n_deletes > 0) {
            const removed = self.idx.removeByPaths(deletes[0..n_deletes]);
            if (removed > 0) {
                log("- {d} files ({d} docs removed)\n", .{ n_deletes, removed });
                updated += removed;
            }
        }

        if (updated > 0) self.dirty = true;
        self.files_changed += updated;
        return updated;
    }

    pub fn deinit(self: *ServerState) void {
        if (self.watcher) |*w| w.deinit();
    }
};

fn log(comptime fmt: []const u8, args: anytype) void {
    std.fs.File.stderr().deprecatedWriter().print(fmt, args) catch {};
}

pub fn run(allocator: Allocator, state: *ServerState, port: u16) !void {
    const addr = std.net.Address.initIp4(.{ 0, 0, 0, 0 }, port);
    var server = try addr.listen(.{ .reuse_address = true });
    defer server.deinit();

    log("zbed server on :{d} ({d} docs indexed)\n", .{ port, state.idx.count() });
    log("  POST /index /search /similarity /rebuild  GET /status\n", .{});

    if (state.root_path.len > 0) {
        state.startWatching() catch |err| {
            log("watch setup failed: {} (reindex via POST /rebuild)\n", .{err});
        };
    }

    // Use poll on the listening socket so we can check inotify between accepts
    const listen_fd = server.stream.handle;
    var poll_fds = [_]std.posix.pollfd{
        .{ .fd = listen_fd, .events = std.posix.POLL.IN, .revents = 0 },
    };

    while (true) {
        // Process file watch events
        _ = state.processFileEvents();
        state.maybeAutoSave();

        // Poll with 100ms timeout so we check inotify regularly
        const poll_result = std.posix.poll(&poll_fds, 100) catch 0;
        if (poll_result == 0) continue; // timeout, loop back to check inotify

        if (poll_fds[0].revents & std.posix.POLL.IN != 0) {
            const conn = server.accept() catch continue;
            handleConn(allocator, state, conn) catch {};
            conn.stream.close();
        }
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
        try handleSimilarity(state, &req);
    } else if (std.mem.eql(u8, target, "/rebuild") and req.head.method == .POST) {
        try handleRebuild(allocator, state, &req);
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
    if (indexed > 0) state.dirty = true;

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

    const group: bool = if (parsed.value.object.get("group")) |gv| switch (gv) {
        .bool => |b| b,
        else => false,
    } else false;

    if (state.idx.count() == 0) return respondJson(req, "{\"results\":[],\"latency_us\":0}");

    var timer = try std.time.Timer.start();
    var qe = embed_mod.QuantizedEmbedding{};
    if (state.model.embedQuantized(query_str, &qe) == 0) return respondJson(req, "{\"results\":[],\"latency_us\":0}");

    const search_index = state.idx.buildSearchIndex();
    var results: [100]search_mod.SearchResult = undefined;
    const raw_k = if (group) @min(k * 4, 100) else @min(k, 100);
    const n = search_index.search(qe.data[0..state.model.embed_dim], qe.norm, raw_k, 0.0, &results);
    const elapsed_us = timer.read() / 1000;

    var resp: std.ArrayListUnmanaged(u8) = .{};
    defer resp.deinit(allocator);
    const w = resp.writer(allocator);

    if (group) {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const groups = try state.idx.groupResults(arena.allocator(), results[0..n], k, 3);

        try w.print("{{\"groups\":[", .{});
        for (groups, 0..) |g, gi| {
            if (gi > 0) try w.writeByte(',');
            try w.print("{{\"file\":", .{});
            try writeJsonStr(w, g.file_path);
            try w.print(",\"score\":{d:.6},\"matches\":[", .{g.best_score});
            for (g.matches, 0..) |m, mi| {
                if (mi > 0) try w.writeByte(',');
                try w.print("{{\"line\":{d},\"text\":", .{m.line_num});
                try writeJsonStr(w, m.content);
                try w.print(",\"score\":{d:.6}}}", .{m.score});
            }
            try w.print("]}}", .{});
        }
        try w.print("],\"latency_us\":{d}}}", .{elapsed_us});
    } else {
        try w.print("{{\"results\":[", .{});
        for (results[0..n], 0..) |r, i| {
            if (i > 0) try w.writeByte(',');
            const doc = state.idx.documents.items[r.doc_idx];
            try w.print("{{\"file\":", .{});
            try writeJsonStr(w, doc.file_path);
            try w.print(",\"line\":{d},\"text\":", .{doc.line_num});
            try writeJsonStr(w, doc.content);
            try w.print(",\"similarity\":{d:.6}}}", .{r.score});
        }
        try w.print("],\"latency_us\":{d}}}", .{elapsed_us});
    }
    try respondJson(req, resp.items);
}

fn handleSimilarity(state: *ServerState, req: *std.http.Server.Request) !void {
    const body = readBody(state.allocator, req) catch {
        return respondErr(req, .bad_request, "{\"error\":\"read failed\"}");
    };
    defer if (body.len > 0) state.allocator.free(body);

    const parsed = std.json.parseFromSlice(std.json.Value, state.allocator, body, .{
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

fn handleRebuild(allocator: Allocator, state: *ServerState, req: *std.http.Server.Request) !void {
    if (state.root_path.len == 0) return respondErr(req, .bad_request, "{\"error\":\"no root path\"}");

    var timer = try std.time.Timer.start();
    state.idx.reset();
    try index_mod.walkAndIndex(allocator, state.root_path, state.model, state.idx, state.walk_options, null);
    state.idx.save(state.root_path) catch {};
    state.last_save_ns = @intCast(std.time.nanoTimestamp());
    state.dirty = false;
    const elapsed_ms = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
    state.files_changed = 0;

    log("rebuild: {d} docs in {d:.1}ms\n", .{ state.idx.count(), elapsed_ms });

    var buf: [256]u8 = undefined;
    const resp = std.fmt.bufPrint(&buf, "{{\"documents\":{d},\"rebuild_ms\":{d:.1}}}", .{ state.idx.count(), elapsed_ms }) catch "{\"rebuilt\":true}";
    try respondJson(req, resp);
}

fn handleStatus(state: *ServerState, req: *std.http.Server.Request) !void {
    var buf: [512]u8 = undefined;
    const resp = std.fmt.bufPrint(&buf, "{{\"documents\":{d},\"dimensions\":{d},\"watching\":{s},\"files_changed\":{d}}}", .{
        state.idx.count(),
        state.idx.dim,
        if (state.watcher != null) "true" else "false",
        state.files_changed,
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
