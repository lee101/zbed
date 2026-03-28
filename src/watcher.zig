const std = @import("std");
const posix = std.posix;
const linux = std.os.linux;
const Allocator = std.mem.Allocator;

pub const Event = struct {
    rel_path: []const u8,
    kind: Kind,
    pub const Kind = enum { created, modified, deleted };
};

pub const Watcher = struct {
    fd: i32,
    wd_map: std.AutoHashMap(i32, []const u8),
    root: []const u8,
    allocator: Allocator,
    debounce_ns: u64 = 200_000_000,
    pending: std.StringHashMap(PendingEvent),

    const PendingEvent = struct {
        kind: Event.Kind,
        timestamp: u64,
    };

    pub fn init(allocator: Allocator, root: []const u8) !Watcher {
        const fd = try posix.inotify_init1(linux.IN.NONBLOCK | linux.IN.CLOEXEC);
        var w = Watcher{
            .fd = fd,
            .wd_map = std.AutoHashMap(i32, []const u8).init(allocator),
            .root = root,
            .allocator = allocator,
            .pending = std.StringHashMap(PendingEvent).init(allocator),
        };
        try w.watchRecursive(root, "");
        return w;
    }

    pub fn deinit(self: *Watcher) void {
        var it = self.wd_map.iterator();
        while (it.next()) |entry| {
            posix.inotify_rm_watch(self.fd, entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.wd_map.deinit();
        var pit = self.pending.iterator();
        while (pit.next()) |entry| self.allocator.free(entry.key_ptr.*);
        self.pending.deinit();
        posix.close(self.fd);
    }

    fn watchRecursive(self: *Watcher, root: []const u8, rel_prefix: []const u8) !void {
        var path_buf: [4096]u8 = undefined;
        const dir_path = if (rel_prefix.len > 0)
            try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ root, rel_prefix })
        else
            root;

        self.addWatch(dir_path, rel_prefix) catch {};

        var dir = std.fs.cwd().openDir(dir_path, .{ .iterate = true }) catch return;
        defer dir.close();
        var iter = dir.iterate();
        while (iter.next() catch null) |entry| {
            if (entry.kind != .directory) continue;
            if (isIgnoredDir(entry.name)) continue;
            var sub_buf: [4096]u8 = undefined;
            const sub_rel = if (rel_prefix.len > 0)
                std.fmt.bufPrint(&sub_buf, "{s}/{s}", .{ rel_prefix, entry.name }) catch continue
            else
                std.fmt.bufPrint(&sub_buf, "{s}", .{entry.name}) catch continue;
            self.watchRecursive(root, sub_rel) catch continue;
        }
    }

    fn addWatch(self: *Watcher, path: []const u8, rel: []const u8) !void {
        const mask: u32 = linux.IN.CREATE | linux.IN.MODIFY | linux.IN.DELETE |
            linux.IN.MOVED_FROM | linux.IN.MOVED_TO | linux.IN.CLOSE_WRITE;
        const wd = try posix.inotify_add_watch(self.fd, path, mask);
        const owned_rel = try self.allocator.dupe(u8, rel);
        try self.wd_map.put(wd, owned_rel);
    }

    pub fn poll(self: *Watcher, out: []Event) usize {
        var buf: [8192]u8 align(@alignOf(linux.inotify_event)) = undefined;
        const n = posix.read(self.fd, &buf) catch |err| switch (err) {
            error.WouldBlock => return self.flushPending(out),
            else => return 0,
        };
        if (n == 0) return self.flushPending(out);

        var offset: usize = 0;
        while (offset + @sizeOf(linux.inotify_event) <= n) {
            const ev: *const linux.inotify_event = @alignCast(@ptrCast(&buf[offset]));
            offset += @sizeOf(linux.inotify_event) + ev.len;

            const name = if (ev.getName()) |s| s else continue;
            if (name.len == 0) continue;

            const dir_rel = self.wd_map.get(ev.wd) orelse continue;
            var path_buf: [4096]u8 = undefined;
            const rel = if (dir_rel.len > 0)
                std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ dir_rel, name }) catch continue
            else
                std.fmt.bufPrint(&path_buf, "{s}", .{name}) catch continue;

            const kind: Event.Kind = if (ev.mask & (linux.IN.DELETE | linux.IN.MOVED_FROM) != 0)
                .deleted
            else if (ev.mask & linux.IN.CREATE != 0)
                .created
            else
                .modified;

            if (ev.mask & linux.IN.CREATE != 0 and ev.mask & linux.IN.ISDIR != 0) {
                self.watchRecursive(self.root, rel) catch {};
            }

            const now = @as(u64, @intCast(std.time.nanoTimestamp()));
            const owned_key = self.allocator.dupe(u8, rel) catch continue;
            if (self.pending.fetchPut(owned_key, .{ .kind = kind, .timestamp = now }) catch null) |old| {
                self.allocator.free(old.key);
            }
        }

        return self.flushPending(out);
    }

    fn flushPending(self: *Watcher, out: []Event) usize {
        const now = @as(u64, @intCast(std.time.nanoTimestamp()));
        var count: usize = 0;
        var to_remove: [256][]const u8 = undefined;
        var remove_count: usize = 0;

        var it = self.pending.iterator();
        while (it.next()) |entry| {
            if (now - entry.value_ptr.timestamp < self.debounce_ns) continue;
            if (count >= out.len) break;
            out[count] = .{ .rel_path = entry.key_ptr.*, .kind = entry.value_ptr.kind };
            count += 1;
            if (remove_count < to_remove.len) {
                to_remove[remove_count] = entry.key_ptr.*;
                remove_count += 1;
            }
        }

        for (to_remove[0..remove_count]) |key| {
            if (self.pending.fetchRemove(key)) |removed| {
                self.allocator.free(removed.key);
            }
        }

        return count;
    }
};

fn isIgnoredDir(name: []const u8) bool {
    const ignored = [_][]const u8{ ".git", ".zbed", ".bed", "node_modules", ".cache", "__pycache__", ".venv", "model", "zig-out", ".zig-cache" };
    for (ignored) |ign| {
        if (std.mem.eql(u8, name, ign)) return true;
    }
    return false;
}
