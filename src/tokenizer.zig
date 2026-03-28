const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Tokenizer = struct {
    vocab: std.StringHashMapUnmanaged(i16),
    id_to_token: std.AutoHashMapUnmanaged(i16, []const u8),
    cls_id: i16,
    sep_id: i16,
    unk_id: i16,
    add_special: bool,
    allocator: Allocator,
    owned_keys: std.ArrayListUnmanaged([]u8),

    pub fn init(allocator: Allocator) Tokenizer {
        return .{
            .vocab = .{},
            .id_to_token = .{},
            .cls_id = -1,
            .sep_id = -1,
            .unk_id = -1,
            .add_special = true,
            .allocator = allocator,
            .owned_keys = .{},
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        for (self.owned_keys.items) |key| {
            self.allocator.free(key);
        }
        self.owned_keys.deinit(self.allocator);
        self.vocab.deinit(self.allocator);
        self.id_to_token.deinit(self.allocator);
    }

    pub fn loadFromFile(self: *Tokenizer, path: []const u8) !void {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const data = try file.readToEndAlloc(self.allocator, 64 * 1024 * 1024);
        defer self.allocator.free(data);

        try self.loadFromJson(data);
    }

    pub fn loadFromJson(self: *Tokenizer, json_bytes: []const u8) !void {
        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, json_bytes, .{
            .allocate = .alloc_always,
            .max_value_len = null,
        });
        defer parsed.deinit();

        const root = parsed.value;
        const model_val = root.object.get("model") orelse return error.MissingModelField;
        const vocab_val = model_val.object.get("vocab") orelse return error.MissingVocabField;

        const vocab_obj = vocab_val.object;

        var it = vocab_obj.iterator();
        while (it.next()) |entry| {
            const token_str = entry.key_ptr.*;
            const id_val = entry.value_ptr.*;
            const id_int = switch (id_val) {
                .integer => |v| v,
                else => continue,
            };
            if (id_int < 0 or id_int >= 32768) continue;
            const id: i16 = @intCast(id_int);

            const owned = try self.allocator.dupe(u8, token_str);
            try self.owned_keys.append(self.allocator, owned);
            try self.vocab.put(self.allocator, owned, id);
            try self.id_to_token.put(self.allocator, id, owned);
        }

        if (self.vocab.get("[CLS]")) |id| self.cls_id = id;
        if (self.vocab.get("[SEP]")) |id| self.sep_id = id;
        if (self.vocab.get("[UNK]")) |id| self.unk_id = id;
    }

    pub fn tokenize(self: *const Tokenizer, text: []const u8, out: []i16) usize {
        var count: usize = 0;
        const max_len = out.len;

        if (self.add_special and self.cls_id >= 0 and count < max_len) {
            out[count] = self.cls_id;
            count += 1;
        }

        var lower_buf: [4096]u8 = undefined;
        const lower_len = @min(text.len, lower_buf.len);
        for (0..lower_len) |i| {
            lower_buf[i] = std.ascii.toLower(text[i]);
        }
        const lower = lower_buf[0..lower_len];

        var word_start: ?usize = null;
        for (lower, 0..) |ch, i| {
            const is_word = isWordChar(ch);
            if (is_word) {
                if (word_start == null) word_start = i;
            } else {
                if (word_start) |ws| {
                    count = self.tokenizeWord(lower[ws..i], out, count, max_len);
                    word_start = null;
                }
                if (!isSpace(ch)) {
                    if (count < max_len) {
                        if (self.vocab.get(lower[i .. i + 1])) |id| {
                            out[count] = id;
                            count += 1;
                        }
                    }
                }
            }
        }
        if (word_start) |ws| {
            count = self.tokenizeWord(lower[ws..lower_len], out, count, max_len);
        }

        if (self.add_special and self.sep_id >= 0 and count < max_len) {
            out[count] = self.sep_id;
            count += 1;
        }

        return count;
    }

    fn tokenizeWord(self: *const Tokenizer, word: []const u8, out: []i16, start_count: usize, max_len: usize) usize {
        var count = start_count;
        if (word.len == 0 or count >= max_len) return count;

        if (self.vocab.get(word)) |id| {
            out[count] = id;
            return count + 1;
        }

        var pos: usize = 0;
        var sub_buf: [256]u8 = undefined;

        while (pos < word.len and count < max_len) {
            var end = word.len;
            var matched: i16 = -1;

            while (end > pos) {
                const piece = word[pos..end];
                if (pos > 0) {
                    if (piece.len + 2 <= sub_buf.len) {
                        sub_buf[0] = '#';
                        sub_buf[1] = '#';
                        @memcpy(sub_buf[2 .. 2 + piece.len], piece);
                        const sub = sub_buf[0 .. 2 + piece.len];
                        if (self.vocab.get(sub)) |id| {
                            matched = id;
                            break;
                        }
                    }
                } else {
                    if (self.vocab.get(piece)) |id| {
                        matched = id;
                        break;
                    }
                }
                end -= 1;
            }

            if (matched == -1) {
                if (self.unk_id >= 0 and count < max_len) {
                    out[count] = self.unk_id;
                    count += 1;
                }
                return count;
            }

            out[count] = matched;
            count += 1;
            pos = end;
        }

        return count;
    }

    pub fn vocabSize(self: *const Tokenizer) usize {
        return self.vocab.count();
    }
};

fn isWordChar(ch: u8) bool {
    return std.ascii.isAlphanumeric(ch) or ch == '\'' or ch == '_';
}

fn isSpace(ch: u8) bool {
    return ch == ' ' or ch == '\t' or ch == '\n' or ch == '\r';
}

test "basic tokenization without model" {
    const allocator = std.testing.allocator;
    var tok = Tokenizer.init(allocator);
    defer tok.deinit();

    const words = [_][]const u8{ "hello", "world", "##lo", "hel", "[CLS]", "[SEP]", "[UNK]" };
    const ids = [_]i16{ 0, 1, 2, 3, 4, 5, 6 };
    for (words, ids) |w, id| {
        const owned = try allocator.dupe(u8, w);
        try tok.owned_keys.append(allocator, owned);
        try tok.vocab.put(allocator, owned, id);
    }
    tok.cls_id = 4;
    tok.sep_id = 5;
    tok.unk_id = 6;

    var out: [64]i16 = undefined;
    const n = tok.tokenize("hello world", &out);

    try std.testing.expect(n >= 3);
    try std.testing.expectEqual(@as(i16, 4), out[0]);
    try std.testing.expectEqual(@as(i16, 0), out[1]);
    try std.testing.expectEqual(@as(i16, 1), out[2]);
    try std.testing.expectEqual(@as(i16, 5), out[n - 1]);
}

test "tokenize empty and whitespace-only input" {
    const allocator = std.testing.allocator;
    var tok = Tokenizer.init(allocator);
    defer tok.deinit();

    const words_data = [_][]const u8{ "[CLS]", "[SEP]", "[UNK]" };
    const ids_data = [_]i16{ 0, 1, 2 };
    for (words_data, ids_data) |w, id| {
        const owned = try allocator.dupe(u8, w);
        try tok.owned_keys.append(allocator, owned);
        try tok.vocab.put(allocator, owned, id);
    }
    tok.cls_id = 0;
    tok.sep_id = 1;
    tok.unk_id = 2;

    var out: [64]i16 = undefined;
    const n1 = tok.tokenize("", &out);
    try std.testing.expectEqual(@as(usize, 2), n1);
    try std.testing.expectEqual(@as(i16, 0), out[0]);
    try std.testing.expectEqual(@as(i16, 1), out[1]);

    const n2 = tok.tokenize("   \t\n  ", &out);
    try std.testing.expectEqual(@as(usize, 2), n2);
}

test "tokenize with punctuation" {
    const allocator = std.testing.allocator;
    var tok = Tokenizer.init(allocator);
    defer tok.deinit();
    tok.add_special = false;

    const words_data = [_][]const u8{ "hello", ".", "!" };
    const ids_data = [_]i16{ 10, 11, 12 };
    for (words_data, ids_data) |w, id| {
        const owned = try allocator.dupe(u8, w);
        try tok.owned_keys.append(allocator, owned);
        try tok.vocab.put(allocator, owned, id);
    }
    tok.unk_id = -1;

    var out: [64]i16 = undefined;
    const n = tok.tokenize("hello!", &out);
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expectEqual(@as(i16, 10), out[0]);
    try std.testing.expectEqual(@as(i16, 12), out[1]);
}

test "wordpiece subword splitting" {
    const allocator = std.testing.allocator;
    var tok = Tokenizer.init(allocator);
    defer tok.deinit();
    tok.add_special = false;

    const words = [_][]const u8{ "play", "##ing" };
    const ids = [_]i16{ 10, 11 };
    for (words, ids) |w, id| {
        const owned = try allocator.dupe(u8, w);
        try tok.owned_keys.append(allocator, owned);
        try tok.vocab.put(allocator, owned, id);
    }
    tok.unk_id = -1;

    var out: [64]i16 = undefined;
    const n = tok.tokenize("playing", &out);
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expectEqual(@as(i16, 10), out[0]);
    try std.testing.expectEqual(@as(i16, 11), out[1]);
}
