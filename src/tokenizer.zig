const std = @import("std");
const Allocator = std.mem.Allocator;

/// WordPiece tokenizer compatible with BERT / HuggingFace tokenizer.json.
/// Loads vocabulary from the `model.vocab` field of tokenizer.json.
pub const Tokenizer = struct {
    /// token string → token id
    vocab: std.StringHashMap(i16),
    /// id → token string (for debug / decode)
    id_to_token: std.AutoHashMap(i16, []const u8),
    /// Special token IDs (-1 means absent)
    cls_id: i16,
    sep_id: i16,
    unk_id: i16,
    /// Whether to prepend [CLS] and append [SEP]
    add_special: bool,

    allocator: Allocator,

    /// Owned copy of every token string (lifetime = Tokenizer lifetime).
    owned_keys: std.ArrayList([]u8),

    pub fn init(allocator: Allocator) Tokenizer {
        return .{
            .vocab = std.StringHashMap(i16).init(allocator),
            .id_to_token = std.AutoHashMap(i16, []const u8).init(allocator),
            .cls_id = -1,
            .sep_id = -1,
            .unk_id = -1,
            .add_special = true,
            .allocator = allocator,
            .owned_keys = std.ArrayList([]u8).init(allocator),
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        for (self.owned_keys.items) |key| {
            self.allocator.free(key);
        }
        self.owned_keys.deinit();
        self.vocab.deinit();
        self.id_to_token.deinit();
    }

    /// Load vocabulary from a HuggingFace tokenizer.json file.
    pub fn loadFromFile(self: *Tokenizer, path: []const u8) !void {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const data = try file.readToEndAlloc(self.allocator, 64 * 1024 * 1024);
        defer self.allocator.free(data);

        try self.loadFromJson(data);
    }

    /// Parse tokenizer.json content and populate vocabulary.
    pub fn loadFromJson(self: *Tokenizer, json_bytes: []const u8) !void {
        // We need to find "model" -> "vocab" in the JSON.
        // The HuggingFace tokenizer.json has structure:
        // { "model": { "vocab": { "token": id, ... }, ... }, ... }
        //
        // We use std.json to parse this.
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

            // Make an owned copy of the key
            const owned = try self.allocator.dupe(u8, token_str);
            try self.owned_keys.append(owned);
            try self.vocab.put(owned, id);
            try self.id_to_token.put(id, owned);
        }

        // Resolve special tokens
        if (self.vocab.get("[CLS]")) |id| self.cls_id = id;
        if (self.vocab.get("[SEP]")) |id| self.sep_id = id;
        if (self.vocab.get("[UNK]")) |id| self.unk_id = id;
    }

    /// Tokenize text into token IDs using WordPiece algorithm.
    pub fn tokenize(self: *const Tokenizer, text: []const u8, out: []i16) usize {
        var count: usize = 0;
        const max_len = out.len;

        // Add [CLS]
        if (self.add_special and self.cls_id >= 0 and count < max_len) {
            out[count] = self.cls_id;
            count += 1;
        }

        // Lowercase the text
        var lower_buf: [4096]u8 = undefined;
        const lower_len = @min(text.len, lower_buf.len);
        for (0..lower_len) |i| {
            lower_buf[i] = std.ascii.toLower(text[i]);
        }
        const lower = lower_buf[0..lower_len];

        // Split into words and tokenize each
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
                // Punctuation as single token
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
        // Last word
        if (word_start) |ws| {
            count = self.tokenizeWord(lower[ws..lower_len], out, count, max_len);
        }

        // Add [SEP]
        if (self.add_special and self.sep_id >= 0 and count < max_len) {
            out[count] = self.sep_id;
            count += 1;
        }

        return count;
    }

    /// WordPiece tokenization for a single word.
    fn tokenizeWord(self: *const Tokenizer, word: []const u8, out: []i16, start_count: usize, max_len: usize) usize {
        var count = start_count;
        if (word.len == 0 or count >= max_len) return count;

        // Try full word first
        if (self.vocab.get(word)) |id| {
            out[count] = id;
            return count + 1;
        }

        // WordPiece: greedy longest-match from left
        var pos: usize = 0;
        var sub_buf: [256]u8 = undefined;

        while (pos < word.len and count < max_len) {
            var end = word.len;
            var matched: i16 = -1;

            while (end > pos) {
                const piece = word[pos..end];
                if (pos > 0) {
                    // Build "##" + piece
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
                // Unknown subword → emit [UNK] and skip entire word
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

// ─── tests ───────────────────────────────────────────────────────────
test "basic tokenization without model" {
    const allocator = std.testing.allocator;
    var tok = Tokenizer.init(allocator);
    defer tok.deinit();

    // Build a tiny vocab
    const words = [_][]const u8{ "hello", "world", "##lo", "hel", "[CLS]", "[SEP]", "[UNK]" };
    const ids = [_]i16{ 0, 1, 2, 3, 4, 5, 6 };
    for (words, ids) |w, id| {
        const owned = try allocator.dupe(u8, w);
        try tok.owned_keys.append(owned);
        try tok.vocab.put(owned, id);
    }
    tok.cls_id = 4;
    tok.sep_id = 5;
    tok.unk_id = 6;

    var out: [64]i16 = undefined;
    const n = tok.tokenize("hello world", &out);

    // Should have [CLS] hello world [SEP]
    try std.testing.expect(n >= 3);
    try std.testing.expectEqual(@as(i16, 4), out[0]); // [CLS]
    try std.testing.expectEqual(@as(i16, 0), out[1]); // hello
    try std.testing.expectEqual(@as(i16, 1), out[2]); // world
    try std.testing.expectEqual(@as(i16, 5), out[n - 1]); // [SEP]
}

test "wordpiece subword splitting" {
    const allocator = std.testing.allocator;
    var tok = Tokenizer.init(allocator);
    defer tok.deinit();
    tok.add_special = false;

    // vocab: "play" -> 10, "##ing" -> 11
    const words = [_][]const u8{ "play", "##ing" };
    const ids = [_]i16{ 10, 11 };
    for (words, ids) |w, id| {
        const owned = try allocator.dupe(u8, w);
        try tok.owned_keys.append(owned);
        try tok.vocab.put(owned, id);
    }
    tok.unk_id = -1;

    var out: [64]i16 = undefined;
    const n = tok.tokenize("playing", &out);
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expectEqual(@as(i16, 10), out[0]); // play
    try std.testing.expectEqual(@as(i16, 11), out[1]); // ##ing
}
